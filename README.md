# TP 1 — Mini-projet : contrôle de l'urbanisation
### De l'alerte satellite au bâtiment : Earth Engine pour repérer, Segment Anything pour délimiter

**Durée :** 1h30 — **Niveau :** aucun prérequis en programmation

## Le besoin
Vous travaillez dans une agence urbaine. On vous demande, pour une commune en forte croissance :
**« Où a-t-on construit ces dernières années, combien de bâtiments sont concernés et quelle surface représentent-ils ? »**

Parcourir toute la commune sur des images très haute résolution prendrait des jours. On procède donc en entonnoir :

| Étape | Outil | Résolution | Question |
|---|---|---|---|
| 2 | **Earth Engine** + carte d'occupation du sol par Deep Learning | 10 m, toute la commune | **Où** le bâti est-il apparu ? Quel secteur inspecter en priorité ? |
| 3 | Archive d'images très haute résolution (Esri Wayback) | ≈ 50 cm, un secteur | À quoi ressemblait le secteur **avant** et **après** ? |
| 4 | **Segment Anything** piloté par un mot | ≈ 50 cm | Où est **chaque bâtiment**, aux deux dates ? |
| 5 | Comparaison des deux dates | bâtiment | Quels bâtiments sont **nouveaux** ? |
| 6 | Fiche de secteur et export SIG | | Que transmet-on au service de contrôle ? |

## Où est l'IA ?
- **Étape 2 :** on exploite une carte produite par un modèle de Deep Learning (Dynamic World) pour lancer une alerte.
- **Étape 4 :** deux modèles de fondation enchaînés. *Grounding DINO* lit votre mot (« building ») et propose des rectangles ; *Segment Anything (SAM)* détoure l'objet dans chaque rectangle. Aucun des deux n'a été entraîné sur nos images.
- **Étape 5 :** pas d'IA, une simple comparaison géométrique. C'est volontaire : un projet réel combine toujours IA et traitements classiques.

---
**Avant de commencer : activer le GPU.** Menu **Exécution → Modifier le type d'exécution → GPU (T4)** puis **Enregistrer**.

Pour exécuter une cellule : cliquer dessus puis **Maj + Entrée**. Exécutez les cellules dans l'ordre.

## Étape 0 — Installation et connexions (≈ 5 minutes)
À la fin de l'installation, redémarrez la session : menu **Exécution → Redémarrer la session**, puis reprenez à la cellule suivante (sans relancer l'installation).

```python
%pip install -q "segment-geospatial[text]" "transformers>=4.50,<5" leafmap geopandas rasterio
```

```python
PROJET = "mon-projet-earth-engine"   # <-- identifiant du projet Earth Engine, à remplacer

import ee
import geemap
ee.Authenticate()
ee.Initialize(project=PROJET)

import torch
print("Earth Engine :", ee.String("connexion réussie").getInfo())
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "ABSENT - menu Exécution > Modifier le type d'exécution > GPU (T4)")
```

```python
import io, re, math, hashlib, datetime
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image as PILImage
import rasterio
from rasterio.plot import show
from rasterio.transform import from_origin
import folium
from IPython.display import display
```

## Étape 1 — La commune à surveiller
Par défaut : **Dar Bouazza**, au sud-ouest de Casablanca, une commune littorale en forte croissance.
Pour une autre commune, modifiez `NOM_COMMUNE` et `COMMUNE_BBOX` (un rectangle de 5 à 15 km de côté).

```python
NOM_COMMUNE = "Dar Bouazza"
COMMUNE_BBOX = [-7.87, 33.48, -7.76, 33.55]     # [ouest, sud, est, nord]
ANNEE_AVANT, ANNEE_APRES = 2016, 2025           # années comparées dans Earth Engine

commune = ee.Geometry.Rectangle(COMMUNE_BBOX)
CENTRE = [(COMMUNE_BBOX[1] + COMMUNE_BBOX[3]) / 2, (COMMUNE_BBOX[0] + COMMUNE_BBOX[2]) / 2]
print(f"{NOM_COMMUNE} : {commune.area(maxError=10).divide(10000).getInfo():,.0f} ha à surveiller")
```

## Étape 2 — L'alerte : où le bâti est-il apparu ? (Earth Engine, 10 m)

> **Étape IA — Exploiter un produit de Deep Learning.** *Dynamic World* est un modèle de Deep Learning qui classe chaque image Sentinel-2 (10 m) et donne, pour chaque pixel, une **probabilité** d'être du bâti. On fait la moyenne de cette probabilité sur une année, puis on compare deux années : un pixel devenu « bâti » entre les deux déclenche une alerte.
>
> À 10 m, on ne voit pas les bâtiments un par un. Mais on couvre toute la commune en quelques secondes : c'est un outil de **ciblage**, pas de mesure.

```python
dynamic_world = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(commune)

def bati(annee):
    proba = dynamic_world.filterDate(f"{annee}-01-01", f"{annee + 1}-01-01").select("built").mean()
    return proba.gt(0.5)

nouveau_bati = bati(ANNEE_APRES).And(bati(ANNEE_AVANT).Not()).unmask(0).clip(commune)

def surface_ha(masque):
    r = (ee.Image.pixelArea().divide(10000).updateMask(masque)
         .reduceRegion(ee.Reducer.sum(), commune, scale=10, maxPixels=1e10, tileScale=4).getInfo())
    return list(r.values())[0] or 0

print(f"Bâti en {ANNEE_AVANT} : {surface_ha(bati(ANNEE_AVANT).clip(commune)):,.0f} ha")
print(f"Bâti en {ANNEE_APRES} : {surface_ha(bati(ANNEE_APRES).clip(commune)):,.0f} ha")
print(f"Alerte « nouveau bâti » {ANNEE_AVANT}-{ANNEE_APRES} : {surface_ha(nouveau_bati):,.0f} ha")
```

### Quel secteur inspecter en priorité ?
Le programme propose un secteur : il calcule, autour de chaque point, la part de pixels en alerte dans un carré de 600 m, et retient l'endroit où cette densité est la plus forte (**cadre bleu**).

**Vous pouvez choisir un autre secteur : cliquez sur la carte.** Un **cadre jaune** de 500 m de côté se place autour du point cliqué. Chaque nouveau clic déplace le cadre. Le bouton « Revenir au secteur proposé » annule votre choix.

Conseil : visez une zone où les taches rouges sont nombreuses, et zoomez sur le fond satellite pour vérifier qu'il s'agit bien d'un quartier récent.

```python
import ipywidgets as widgets
from ipyleaflet import Rectangle
try:   # nécessaire dans Colab pour que la carte réagisse aux clics
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

DEMI_COTE = 250              # demi-côté du secteur inspecté, en mètres

def carre_autour(lat, lon):
    """Rectangle [ouest, sud, est, nord] de 2 x DEMI_COTE mètres de côté, centré sur le point."""
    d_lat = DEMI_COTE / 111320
    d_lon = DEMI_COTE / (111320 * math.cos(math.radians(lat)))
    return [lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat]

# Secteur proposé automatiquement : là où la densité d'alerte est la plus forte
densite = nouveau_bati.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(300, "meters")).rename("densite")
r = (densite.addBands(ee.Image.pixelLonLat())
     .reduceRegion(ee.Reducer.max(3), commune, scale=50, maxPixels=1e9, tileScale=4).getInfo())
part_auto, lon_auto, lat_auto = [r[k] for k in sorted(r)]
print(f"Secteur proposé : latitude {lat_auto:.5f}, longitude {lon_auto:.5f} "
      f"({part_auto:.0%} des pixels alentour sont en alerte)")

carte = geemap.Map(center=CENTRE, zoom=13, height="600px")
carte.add_basemap("SATELLITE")
carte.addLayer(densite.selfMask(), {"min": 0, "max": 0.6, "palette": ["ffffb2", "fd8d3c", "bd0026"]},
               "Densité d'alerte", True, 0.6)
carte.addLayer(nouveau_bati.selfMask(), {"palette": ["#ff1744"]},
               f"Alerte nouveau bâti {ANNEE_AVANT}-{ANNEE_APRES}", True, 0.8)
carte.addLayer(ee.Image().paint(ee.Geometry.Rectangle(carre_autour(lat_auto, lon_auto)), 0, 3),
               {"palette": ["#00e5ff"]}, "Secteur proposé")

# Choix par clic
choix = {"point": None, "cadre": None}
message = widgets.HTML("<b>Secteur retenu :</b> celui proposé (cadre bleu). Cliquez sur la carte pour en choisir un autre.")
retour = widgets.Button(description="Revenir au secteur proposé", icon="undo", layout=widgets.Layout(width="250px"))

def au_clic(**evenement):
    if evenement.get("type") != "click":
        return
    lat, lon = evenement["coordinates"]
    o, s, e, n = carre_autour(lat, lon)
    if choix["cadre"] is None:
        choix["cadre"] = Rectangle(bounds=((s, o), (n, e)), color="#ffea00", weight=3, fill_opacity=0.05)
        carte.add(choix["cadre"])
    else:
        choix["cadre"].bounds = ((s, o), (n, e))
    choix["point"] = (lat, lon)
    message.value = f"<b>Secteur retenu :</b> votre clic, latitude {lat:.5f}, longitude {lon:.5f} (cadre jaune)."

def au_retour(_):
    if choix["cadre"] is not None:
        try:
            carte.remove(choix["cadre"])
        except Exception:
            pass
    choix["point"], choix["cadre"] = None, None
    message.value = "<b>Secteur retenu :</b> celui proposé (cadre bleu). Cliquez sur la carte pour en choisir un autre."

if hasattr(carte, "on_interaction"):
    carte.on_interaction(au_clic)
    retour.on_click(au_retour)
    display(widgets.HBox([retour, message]))
else:
    print("Cette carte ne gère pas les clics : utilisez SECTEUR_MANUEL dans la cellule suivante.")
carte
```

Exécutez la cellule suivante **après** avoir fait votre choix. Elle fixe le secteur pour toute la suite du TP.

*Si le clic ne fonctionne pas dans votre navigateur (aucun cadre jaune n'apparaît), écrivez les coordonnées à la main dans `SECTEUR_MANUEL`, par exemple `(33.512, -7.815)`.*

```python
SECTEUR_MANUEL = None        # secours : (latitude, longitude)

if choix["point"] is not None:
    (lat_s, lon_s), origine = choix["point"], "choisi par un clic"
elif SECTEUR_MANUEL is not None:
    (lat_s, lon_s), origine = SECTEUR_MANUEL, "saisi à la main"
else:
    (lat_s, lon_s), origine = (lat_auto, lon_auto), "proposé automatiquement"

SECTEUR_BBOX = carre_autour(lat_s, lon_s)
secteur = ee.Geometry.Rectangle(SECTEUR_BBOX)
part = nouveau_bati.reduceRegion(ee.Reducer.mean(), secteur, scale=10, maxPixels=1e9).getInfo()
part = list(part.values())[0] or 0
print(f"Secteur {origine} : latitude {lat_s:.5f}, longitude {lon_s:.5f}")
print(f"Taille : {2 * DEMI_COTE} m x {2 * DEMI_COTE} m, dont {part:.0%} de pixels en alerte « nouveau bâti ».")
if part < 0.05:
    print("Attention : très peu d'alertes dans ce secteur, la comparaison avant/après risque d'être peu parlante.")
```

**Avant de continuer, regardez la carte de l'étape précédente.** Les alertes rouges correspondent-elles toutes à des constructions ? À 10 m, un chantier, un sol décapé ou une carrière peuvent aussi déclencher l'alerte : d'où la nécessité de zoomer.

## Étape 3 — Zoomer : les images très haute résolution, avant et après

Le fond satellite d'Esri est mis à jour régulièrement, et chaque ancienne version est conservée dans une archive appelée **Wayback** (depuis 2014). On peut donc retrouver, pour un même lieu, des images prises à des dates différentes.

Le programme :
1. télécharge la liste des versions ;
2. repère celles où l'image de **notre secteur** a réellement changé ;
3. interroge, quand c'est possible, la **date de prise de vue** de chacune.

*Attention : la date de publication d'une version n'est pas la date de prise de vue. L'image a été prise avant, parfois plusieurs mois ou années plus tôt.*

```python
CONFIG_WAYBACK = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
ORIGINE = 20037508.342789244          # demi-périmètre de la projection des cartes web (EPSG:3857)

def tuile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def adresse_tuile(version, z, x, y):
    return version["url"].replace("{level}", str(z)).replace("{row}", str(y)).replace("{col}", str(x))

def lire_tuile(version, z, x, y):
    try:
        r = requests.get(adresse_tuile(version, z, x, y), timeout=30)
        return r.content if r.status_code == 200 and len(r.content) > 500 else None
    except Exception:
        return None

def lister_versions():
    config = requests.get(CONFIG_WAYBACK, timeout=60).json()
    versions = []
    for numero, v in config.items():
        date = re.search(r"(\d{4}-\d{2}-\d{2})", v.get("itemTitle", ""))
        if date and "itemURL" in v:
            versions.append({"numero": int(numero), "publication": date.group(1),
                             "url": v["itemURL"], "meta": v.get("metadataLayerUrl")})
    return sorted(versions, key=lambda v: v["publication"])

def versions_distinctes(versions, lat, lon, z=17):
    """Garde les versions où la tuile du secteur est différente de la précédente."""
    x, y = tuile(lat, lon, z)
    with ThreadPoolExecutor(max_workers=16) as ex:
        contenus = list(ex.map(lambda v: lire_tuile(v, z, x, y), versions))
    distinctes, deja_vues = [], set()
    for v, contenu in zip(versions, contenus):
        if contenu is None:
            continue
        empreinte = hashlib.md5(contenu).hexdigest()
        if empreinte not in deja_vues:
            deja_vues.add(empreinte)
            distinctes.append(v)
    return distinctes

def date_prise_de_vue(version, lat, lon):
    """Interroge le service de métadonnées d'Esri. Renvoie (date, fournisseur, résolution) ou des None."""
    if not version.get("meta"):
        return None, None, None
    parametres = {"f": "json", "where": "1=1", "outFields": "*", "returnGeometry": "false",
                  "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": "4326",
                  "spatialRel": "esriSpatialRelIntersects"}
    for couche in (6, 5, 7, 4, 8, 3):
        try:
            r = requests.get(f"{version['meta']}/{couche}/query", params=parametres, timeout=20).json()
            if r.get("features"):
                a = r["features"][0]["attributes"]
                date = next((a[k] for k in a if "DATE" in k.upper() and isinstance(a[k], (int, float)) and a[k] > 1e11), None)
                if date:
                    date = datetime.datetime.utcfromtimestamp(date / 1000).strftime("%Y-%m-%d")
                return date, a.get("NICE_DESC") or a.get("SRC_DESC"), a.get("SAMP_RES")
        except Exception:
            pass
    return None, None, None

toutes = lister_versions()
print(f"{len(toutes)} versions du fond satellite dans l'archive, de {toutes[0]['publication']} à {toutes[-1]['publication']}.")
distinctes = versions_distinctes(toutes, lat_s, lon_s)
for v in distinctes:
    v["prise_de_vue"], v["fournisseur"], v["resolution"] = date_prise_de_vue(v, lat_s, lon_s)

tableau_versions = pd.DataFrame([{"Numéro de version": v["numero"], "Publiée le": v["publication"],
                                  "Prise de vue": v["prise_de_vue"] or "non renseignée",
                                  "Fournisseur": v["fournisseur"] or "", "Résolution (m)": v["resolution"] or ""}
                                 for v in distinctes])
print(f"{len(distinctes)} images différentes de notre secteur :")
tableau_versions
```

### Télécharger l'image « avant » et l'image « après »
Par défaut, on prend la plus ancienne et la plus récente du tableau. Pour choisir d'autres dates, écrivez le **numéro de version** dans `VERSION_AVANT` ou `VERSION_APRES`.

```python
VERSION_AVANT = None      # ou un numéro de version du tableau
VERSION_APRES = None
ZOOM = 18                 # 18 ≈ 50 cm par pixel ; 19 ≈ 25 cm (pas toujours disponible dans les versions anciennes)

def trouver(numero, defaut):
    if numero is None:
        return defaut
    return next(v for v in toutes if v["numero"] == numero)

def telecharger_image(version, bbox, z, fichier):
    """Assemble les tuiles couvrant le secteur et enregistre un GeoTIFF géoréférencé."""
    x0, y1 = tuile(bbox[1], bbox[0], z)      # coin sud-ouest
    x1, y0 = tuile(bbox[3], bbox[2], z)      # coin nord-est
    cases = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(max_workers=12) as ex:
        contenus = list(ex.map(lambda c: lire_tuile(version, z, *c), cases))
    mosaique = PILImage.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    manquantes = 0
    for (x, y), contenu in zip(cases, contenus):
        if contenu is None:
            manquantes += 1
            continue
        mosaique.paste(PILImage.open(io.BytesIO(contenu)).convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
    pas = 2 * ORIGINE / (256 * 2 ** z)
    transfo = from_origin(-ORIGINE + x0 * 256 * pas, ORIGINE - y0 * 256 * pas, pas, pas)
    tableau = np.moveaxis(np.array(mosaique), -1, 0)
    with rasterio.open(fichier, "w", driver="GTiff", height=tableau.shape[1], width=tableau.shape[2],
                       count=3, dtype="uint8", crs="EPSG:3857", transform=transfo) as sortie:
        sortie.write(tableau)
    return manquantes / len(cases)

avant, apres = trouver(VERSION_AVANT, distinctes[0]), trouver(VERSION_APRES, distinctes[-1])
for nom, version, fichier in [("avant", avant, "image_avant.tif"), ("après", apres, "image_apres.tif")]:
    manque = telecharger_image(version, SECTEUR_BBOX, ZOOM, fichier)
    print(f"Image {nom} : version publiée le {version['publication']}, "
          f"prise de vue {version.get('prise_de_vue') or 'non renseignée'}"
          + (f" - ATTENTION : {manque:.0%} de l'image manque, essayez ZOOM = 17" if manque > 0.1 else ""))

def etiquette(version):
    return version.get("prise_de_vue") or f"publiée le {version['publication']}"

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
for ax, fichier, version, nom in [(axes[0], "image_avant.tif", avant, "AVANT"), (axes[1], "image_apres.tif", apres, "APRÈS")]:
    with rasterio.open(fichier) as src:
        show(src, ax=ax)
    ax.set_title(f"{nom} - {etiquette(version)}", fontsize=14)
    ax.set_axis_off()
plt.tight_layout(); plt.show()
```

**Comparez les deux images à l'œil.** Voyez-vous de nouvelles constructions ? Combien, à peu près ? Notez votre estimation : nous la comparerons à celle de l'IA.

## Étape 4 — Délimiter chaque bâtiment avec Segment Anything

> **Étape IA — Deux modèles de fondation enchaînés.**
> 1. **Grounding DINO** reçoit l'image et un mot en anglais. Il a appris, sur des millions d'images légendées, à relier des mots et des objets : il propose un **rectangle** autour de chaque objet qui correspond au mot, avec un score de confiance.
> 2. **Segment Anything (SAM)** reçoit chaque rectangle et **détoure** précisément l'objet qu'il contient. Il a été entraîné sur plus d'un milliard de masques, mais ne sait pas ce qu'est un bâtiment : il sait seulement séparer un objet de son fond.
>
> Ni l'un ni l'autre n'a été entraîné sur des images du Maroc, ni même spécifiquement sur des images satellite. On les utilise tels quels : c'est le principe du modèle de fondation.

Le chargement des modèles (≈ 3 Go) prend 1 à 2 minutes la première fois.

```python
from samgeo.text_sam import LangSAM
sam = LangSAM()
```

### Pourquoi découper l'image en dalles ?

> **Étape IA — Adapter l'image au modèle.** Grounding DINO a été entraîné sur des photos ordinaires, où l'on voit quelques objets de grande taille. Sur une image de quartier entier, il voit des centaines de petits toits : il a alors tendance à entourer **tout le quartier d'un seul rectangle**, et SAM remplit ce rectangle. On obtient une grande tache, pas des bâtiments.
>
> La parade est classique : on découpe l'image en **dalles** de 512 pixels qui se recouvrent, on interroge le modèle dalle par dalle, puis on rassemble les résultats. On écarte aussi tout objet qui occupe une trop grande part de sa dalle : ce n'est pas un bâtiment, c'est un îlot ou un quartier.

Réglages :
- `MOT` : le mot donné au modèle, en anglais (`building`, `house`, `roof`) ;
- `SEUIL` : la confiance minimale. Plus bas, le modèle trouve plus de bâtiments mais fait plus de fausses détections ;
- `PART_MAX_DALLE` : au-delà de cette part de la dalle, l'objet est écarté.

Le traitement prend environ une minute par image.

```python
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

MOT = "building"
SEUIL = 0.24
SURFACE_MIN, SURFACE_MAX = 25, 5000      # en m² : on écarte les objets trop petits ou trop grands
TAILLE_DALLE, RECOUVREMENT = 512, 128    # en pixels
PART_MAX_DALLE = 0.30

def positions(taille_image):
    if taille_image <= TAILLE_DALLE:
        return [0]
    debuts = list(range(0, taille_image - TAILLE_DALLE, TAILLE_DALLE - RECOUVREMENT))
    return debuts + [taille_image - TAILLE_DALLE]

def masques_de_la_dalle(fichier):
    """Un masque (tableau vrai/faux) par objet trouvé dans la dalle."""
    sam.boxes, sam.masks = [], None
    sam.predict(fichier, MOT, box_threshold=SEUIL, text_threshold=SEUIL)
    if sam.masks is None or len(sam.boxes) == 0:
        return []
    m = sam.masks
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    if m.size == 0:
        return []
    return list(m.reshape((-1,) + m.shape[-2:]) > 0)

def detecter_batiments(fichier_image, nom):
    """Renvoie les polygones des bâtiments détectés, avec leur surface en m²."""
    vide = gpd.GeoDataFrame({"surface_m2": []}, geometry=[], crs="EPSG:4326")
    entiers, coupes, ecartes, dalles = [], [], 0, 0
    with rasterio.open(fichier_image) as src:
        H, W, crs = src.height, src.width, src.crs
        for r in positions(H):
            for c in positions(W):
                h, w = min(TAILLE_DALLE, H - r), min(TAILLE_DALLE, W - c)
                fenetre = Window(c, r, w, h)
                transfo = src.window_transform(fenetre)
                with rasterio.open("dalle.tif", "w", driver="GTiff", height=h, width=w, count=src.count,
                                   dtype="uint8", crs=crs, transform=transfo) as dalle:
                    dalle.write(src.read(window=fenetre))
                dalles += 1
                for masque in masques_de_la_dalle("dalle.tif"):
                    if masque.shape != (h, w) or not masque.any():
                        continue
                    if masque.mean() > PART_MAX_DALLE:          # îlot ou quartier entier
                        ecartes += 1
                        continue
                    formes = [shape(g) for g, _ in shapes(masque.astype("uint8"), mask=masque, transform=transfo)]
                    polygone = max(formes, key=lambda p: p.area).buffer(0)
                    touche_un_bord = ((r > 0 and masque[0].any()) or (r + h < H and masque[-1].any())
                                      or (c > 0 and masque[:, 0].any()) or (c + w < W and masque[:, -1].any()))
                    (coupes if touche_un_bord else entiers).append(polygone)
    if not entiers and not coupes:
        print(f"{nom} : aucun bâtiment trouvé pour « {MOT} » ({dalles} dalles).")
        return vide

    utm = gpd.GeoSeries(entiers + coupes, crs=crs).estimate_utm_crs()
    en_metres = lambda liste: list(gpd.GeoSeries(liste, crs=crs).to_crs(utm).buffer(0)) if liste else []
    gardes = []

    def ajouter(p):
        """Ajoute p aux bâtiments retenus si c'est pertinent. Renvoie 1 si p est écarté car trop grand."""
        if p.area > SURFACE_MAX:
            return 1
        if p.area < SURFACE_MIN:
            return 0
        voisins = [q for q in gardes if q.intersects(p)]
        contenus = [q for q in voisins if q.intersection(p).area > 0.8 * q.area]
        if len(contenus) >= 2:
            return 0                                   # p regroupe plusieurs bâtiments déjà retenus
        if any(q.intersection(p).area > 0.6 * q.union(p).area for q in voisins):
            return 0                                   # p est un doublon (même bâtiment vu dans une autre dalle)
        for q in contenus:                             # q n'était qu'un morceau de p : p le remplace
            gardes.remove(q)
        gardes.append(p)
        return 0

    for p in sorted(en_metres(entiers), key=lambda p: p.area):
        ecartes += ajouter(p)
    # bâtiments coupés par le bord d'une dalle : on recolle les morceaux qui se chevauchent
    emprise = unary_union(gardes) if gardes else None
    morceaux = [p for p in en_metres(coupes)
                if emprise is None or p.intersection(emprise).area <= 0.5 * p.area]
    if morceaux:
        recolles = unary_union(morceaux)
        for p in sorted(getattr(recolles, "geoms", [recolles]), key=lambda p: p.area):
            ecartes += ajouter(p)

    polygones = gpd.GeoDataFrame(geometry=[p.simplify(0.5) for p in gardes], crs=utm)
    polygones["surface_m2"] = polygones.geometry.area
    print(f"{nom} : {len(polygones)} bâtiments détectés sur {dalles} dalles, "
          f"{polygones.surface_m2.sum():,.0f} m² au total ({ecartes} objets trop grands écartés).")
    return polygones[["surface_m2", "geometry"]].reset_index(drop=True)

batiments_avant = detecter_batiments("image_avant.tif", "avant")
batiments_apres = detecter_batiments("image_apres.tif", "apres")
```

## Étape 5 — Comparer : quels bâtiments sont nouveaux ?
Ici, pas d'IA : une règle géométrique simple. Un bâtiment de l'image « après » est déclaré **nouveau** si moins de 30 % de sa surface était déjà couverte par un bâtiment sur l'image « avant ».

On tolère 3 m de décalage, car deux images prises sous des angles différents ne se superposent jamais parfaitement.

```python
PART_MAX_DEJA_BATI = 0.30
TOLERANCE = 3     # mètres

def part_couverte(polygones, reference):
    if reference.empty or polygones.empty:
        return pd.Series(0.0, index=polygones.index)
    emprise = reference.to_crs(polygones.crs).geometry.buffer(TOLERANCE).unary_union
    return polygones.geometry.intersection(emprise).area / polygones.geometry.area

batiments_apres["deja_bati"] = part_couverte(batiments_apres, batiments_avant)
batiments_apres["statut"] = np.where(batiments_apres.deja_bati < PART_MAX_DEJA_BATI, "nouveau", "existant")
batiments_avant["retrouve"] = part_couverte(batiments_avant, batiments_apres)

nouveaux = batiments_apres[batiments_apres.statut == "nouveau"]
existants = batiments_apres[batiments_apres.statut == "existant"]
disparus = batiments_avant[batiments_avant.retrouve < PART_MAX_DEJA_BATI]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
with rasterio.open("image_avant.tif") as src:
    show(src, ax=axes[0]); crs_image = src.crs
if not batiments_avant.empty:
    batiments_avant.to_crs(crs_image).boundary.plot(ax=axes[0], color="#ffea00", linewidth=1.5)
axes[0].set_title(f"AVANT - {etiquette(avant)} : {len(batiments_avant)} bâtiments", fontsize=14)
with rasterio.open("image_apres.tif") as src:
    show(src, ax=axes[1])
if not existants.empty:
    existants.to_crs(crs_image).boundary.plot(ax=axes[1], color="#ffea00", linewidth=1.5)
if not nouveaux.empty:
    nouveaux.to_crs(crs_image).plot(ax=axes[1], facecolor="#ff1744", edgecolor="white", alpha=0.6, linewidth=1)
axes[1].set_title(f"APRÈS - {etiquette(apres)} : {len(nouveaux)} nouveaux bâtiments", fontsize=14)
axes[1].legend(handles=[Patch(facecolor="#ff1744", label="Nouveau bâtiment"),
                        Patch(facecolor="none", edgecolor="#ffea00", label="Bâtiment existant")], loc="lower right")
for ax in axes:
    ax.set_axis_off()
plt.tight_layout(); plt.savefig("carte_controle.png", dpi=150, bbox_inches="tight"); plt.show()
```

### Comparer avec un rideau avant / après
**Faites glisser la barre blanche** : à gauche l'image « avant », à droite l'image « après ». Décochez la case pour masquer les contours et juger sur l'image seule.

```python
import base64, uuid
from PIL import ImageDraw
from IPython.display import HTML

def image_annotee(fichier, couches=()):
    """Image du secteur avec des polygones dessinés. couches : (polygones, couleur du trait, remplissage ou None)."""
    with rasterio.open(fichier) as src:
        fond = PILImage.fromarray(np.moveaxis(src.read()[:3], 0, -1)).convert("RGBA")
        t, crs = src.transform, src.crs
    calque = PILImage.new("RGBA", fond.size, (0, 0, 0, 0))
    dessin = ImageDraw.Draw(calque)
    for polygones, trait, remplissage in couches:
        if polygones.empty:
            continue
        for geometrie in polygones.to_crs(crs).geometry:
            for partie in getattr(geometrie, "geoms", [geometrie]):
                if partie.geom_type != "Polygon":
                    continue
                points = [((x - t.c) / t.a, (y - t.f) / t.e) for x, y in partie.exterior.coords]
                if remplissage:
                    dessin.polygon(points, fill=remplissage)
                dessin.line(points + points[:1], fill=trait, width=3)
    return PILImage.alpha_composite(fond, calque).convert("RGB")

def en_texte(image, taille):
    tampon = io.BytesIO()
    image.resize(taille).save(tampon, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(tampon.getvalue()).decode()

def rideau(gauche, droite, gauche_annotee, droite_annotee, titre_gauche, titre_droite, largeur=900):
    """Comparateur d'images à barre glissante, affiché directement dans le notebook."""
    k, taille = uuid.uuid4().hex[:8], gauche.size
    sources = {n: en_texte(im, taille) for n, im in
               [("g0", gauche), ("d0", droite), ("g1", gauche_annotee), ("d1", droite_annotee)]}
    etiquette_css = ("position:absolute;top:10px;padding:4px 10px;background:rgba(0,0,0,.65);color:#fff;"
                     "font:bold 14px sans-serif;border-radius:4px;pointer-events:none")
    return HTML(f"""
<div id="r{k}" style="position:relative;width:100%;max-width:{largeur}px;user-select:none;touch-action:none;cursor:ew-resize">
  <img id="d{k}" src="{sources['d1']}" draggable="false" style="display:block;width:100%">
  <img id="g{k}" src="{sources['g1']}" draggable="false"
       style="position:absolute;top:0;left:0;width:100%;height:100%;clip-path:inset(0 50% 0 0)">
  <div id="b{k}" style="position:absolute;top:0;bottom:0;left:50%;width:4px;margin-left:-2px;background:#fff;box-shadow:0 0 5px #000;pointer-events:none"></div>
  <div id="p{k}" style="position:absolute;top:50%;left:50%;width:40px;height:40px;margin:-20px 0 0 -20px;border-radius:50%;background:#fff;box-shadow:0 0 5px #000;pointer-events:none;text-align:center;font:bold 20px/40px sans-serif;color:#333">&#8596;</div>
  <span style="{etiquette_css};left:10px">{titre_gauche}</span>
  <span style="{etiquette_css};right:10px">{titre_droite}</span>
</div>
<div style="max-width:{largeur}px;margin-top:8px;font:14px sans-serif">
  <label><input type="checkbox" id="c{k}" checked> Afficher les contours des bâtiments</label>
  &nbsp;&nbsp; <span style="color:#ff1744">&#9632;</span> nouveau bâtiment
  &nbsp;&nbsp; <span style="color:#c9b800">&#9632;</span> bâtiment existant
</div>
<script>
(function() {{
  var r = document.getElementById("r{k}"), g = document.getElementById("g{k}"), d = document.getElementById("d{k}");
  var b = document.getElementById("b{k}"), p = document.getElementById("p{k}"), c = document.getElementById("c{k}");
  var src = {{g0: "{sources['g0']}", d0: "{sources['d0']}", g1: g.src, d1: d.src}};
  var glisse = false;
  function placer(e) {{
    var cadre = r.getBoundingClientRect();
    var v = Math.max(0, Math.min(100, 100 * (e.clientX - cadre.left) / cadre.width));
    g.style.clipPath = "inset(0 " + (100 - v) + "% 0 0)";
    b.style.left = v + "%"; p.style.left = v + "%";
  }}
  r.addEventListener("pointerdown", function(e) {{ glisse = true; r.setPointerCapture(e.pointerId); placer(e); }});
  r.addEventListener("pointermove", function(e) {{ if (glisse) placer(e); }});
  r.addEventListener("pointerup", function() {{ glisse = false; }});
  r.addEventListener("pointercancel", function() {{ glisse = false; }});
  c.addEventListener("change", function() {{ g.src = c.checked ? src.g1 : src.g0; d.src = c.checked ? src.d1 : src.d0; }});
}})();
</script>""")

JAUNE, ROUGE = (255, 234, 0, 255), (255, 23, 68, 255)
comparateur = rideau(
    image_annotee("image_avant.tif"),
    image_annotee("image_apres.tif"),
    image_annotee("image_avant.tif", [(batiments_avant, JAUNE, None)]),
    image_annotee("image_apres.tif", [(existants, JAUNE, None), (nouveaux, ROUGE, (255, 23, 68, 110))]),
    f"AVANT - {etiquette(avant)}", f"APRÈS - {etiquette(apres)}")
display(comparateur)
```

**Contrôlez le résultat, bâtiment par bâtiment :**
- Trouvez un bâtiment rouge qui est bien une construction nouvelle.
- Trouvez une **fausse alerte** : un bâtiment qui existait déjà mais que le modèle avait manqué sur l'image « avant ».
- Trouvez un **oubli** : une construction nouvelle que le modèle n'a pas détourée.
- Votre estimation à l'œil de l'étape 3 est-elle proche du chiffre de l'IA ?

Essayez ensuite `MOT = "house"` ou `SEUIL = 0.20` à l'étape 4, puis relancez les étapes 4 et 5.

## Étape 6 — La fiche de secteur
### L'alerte à 10 m avait-elle vu juste ?
On revient à Earth Engine : pour chaque nouveau bâtiment trouvé par SAM, on regarde s'il tombe dans un pixel que l'alerte à 10 m avait signalé.

```python
def dans_alerte(polygones):
    if polygones.empty:
        return pd.Series([], dtype=bool)
    url = nouveau_bati.toByte().getDownloadURL({"region": secteur.buffer(100).bounds(), "scale": 10,
                                                "crs": "EPSG:4326", "format": "GEO_TIFF"})
    with rasterio.open(io.BytesIO(requests.get(url, timeout=300).content)) as src:
        centres = polygones.to_crs(4326).geometry.representative_point()
        valeurs = [v[0] for v in src.sample([(p.x, p.y) for p in centres])]
    return pd.Series(valeurs, index=polygones.index) == 1

batiments_apres["alerte_10m"] = False
if not nouveaux.empty:
    batiments_apres.loc[nouveaux.index, "alerte_10m"] = dans_alerte(nouveaux)
    nouveaux = batiments_apres[batiments_apres.statut == "nouveau"]

surface_secteur_ha = (2 * DEMI_COTE) ** 2 / 10000
print("=" * 62)
print(f"FICHE DE SECTEUR - {NOM_COMMUNE}")
print("=" * 62)
print(f"Centre du secteur      : {lat_s:.5f}, {lon_s:.5f} ({surface_secteur_ha:.0f} ha)")
print(f"Image avant            : {etiquette(avant)}")
print(f"Image après            : {etiquette(apres)}")
print(f"Bâtiments avant        : {len(batiments_avant)}, soit {batiments_avant.surface_m2.sum():,.0f} m² au sol")
print(f"Bâtiments après        : {len(batiments_apres)}, soit {batiments_apres.surface_m2.sum():,.0f} m² au sol")
print(f"NOUVEAUX BÂTIMENTS     : {len(nouveaux)}, soit {nouveaux.surface_m2.sum():,.0f} m² au sol")
print(f"Bâtiments disparus     : {len(disparus)} (démolitions ou bâtiments manqués sur l'image après)")
if len(nouveaux):
    print(f"Confirmés par l'alerte : {int(nouveaux.alerte_10m.sum())} sur {len(nouveaux)} "
          f"({nouveaux.alerte_10m.mean():.0%}) tombent dans un pixel signalé à 10 m")
print("=" * 62)
print("Résultat issu de modèles d'IA, à vérifier avant tout usage administratif.")
```

### Carte interactive et export
Les nouveaux bâtiments sont numérotés. Cliquez sur un bâtiment pour voir sa surface.

```python
resultat = batiments_apres.to_crs(4326).copy()
resultat["surface_m2"] = resultat.surface_m2.round(0)
resultat["deja_bati"] = resultat.deja_bati.round(2)
resultat.insert(0, "id", range(1, len(resultat) + 1))

carte = folium.Map(location=[lat_s, lon_s], zoom_start=17, max_zoom=19,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   attr="Esri World Imagery")
if not resultat.empty:
    folium.GeoJson(
        resultat.to_json(),
        style_function=lambda f: {"color": "#ff1744" if f["properties"]["statut"] == "nouveau" else "#ffea00",
                                  "weight": 2, "fillOpacity": 0.35 if f["properties"]["statut"] == "nouveau" else 0.05},
        tooltip=folium.GeoJsonTooltip(fields=["id", "statut", "surface_m2", "alerte_10m"],
                                      aliases=["Bâtiment", "Statut", "Surface (m²)", "Dans l'alerte 10 m"]),
    ).add_to(carte)

nom = NOM_COMMUNE.lower().replace(" ", "_")
resultat.to_file(f"batiments_{nom}.geojson", driver="GeoJSON")
resultat.drop(columns="geometry").to_csv(f"batiments_{nom}.csv", index=False)
print("Fichiers créés (panneau Fichiers à gauche, clic droit > Télécharger) :")
for f in (f"batiments_{nom}.geojson", f"batiments_{nom}.csv", "carte_controle.png", "image_avant.tif", "image_apres.tif"):
    print(" -", f)
carte
```

## Récapitulatif : ce que chaque outil a apporté

| Outil | Ce qu'il sait faire | Ce qu'il ne sait pas faire |
|---|---|---|
| Earth Engine + Dynamic World (10 m) | Surveiller toute une commune, chaque année, en quelques secondes | Voir un bâtiment isolé, distinguer un chantier d'une construction |
| Archive très haute résolution | Montrer le terrain à deux dates | Fournir une image à la date de son choix |
| Grounding DINO + SAM | Détourer chaque bâtiment sans entraînement | Garantir qu'aucun bâtiment n'est oublié, dater une construction |
| Comparaison géométrique | Lister les nouveaux bâtiments | Corriger les erreurs des étapes précédentes |

## Questions de synthèse (à discuter en groupe)
1. Sur votre secteur, combien de fausses alertes et d'oublis avez-vous repérés ? Lequel des deux est le plus gênant pour un service de contrôle ?
2. Un bâtiment « nouveau » est-il forcément une infraction ? Quelles données faudrait-il croiser (permis de construire, plan d'aménagement, cadastre) ?
3. Les dates des images sont subies, et parfois inconnues. Qu'est-ce que cela change pour la valeur du résultat ? Qu'apporteraient vos propres images (Pléiades, Mohammed VI, orthophotos) ?
4. Les 10 m d'Earth Engine et les 50 cm de SAM ne se contredisent-ils jamais ? Que faire quand SAM trouve des bâtiments nouveaux là où il n'y avait pas d'alerte ?
5. SAM n'a jamais été entraîné sur des images marocaines. Pour quels types de bâti fonctionne-t-il bien ou mal (villas isolées, lotissements denses, habitat rural, hangars) ?
6. Comment passer d'un secteur de 25 ha à toute une province ? Quel serait le rôle de l'agent sur le terrain ?

---
**À retenir**
- Un projet réel enchaîne plusieurs outils : repérer large à basse résolution, puis zoomer là où cela compte.
- Les modèles de fondation donnent un résultat utile sans entraînement, mais ils se trompent : le contrôle humain fait partie du processus.
- Le livrable n'est pas une carte d'IA, c'est une liste de bâtiments à vérifier, avec ses limites écrites noir sur blanc.

*Données : Dynamic World (Google, WRI), Sentinel-2 (Copernicus/ESA), Esri World Imagery et son archive Wayback (Esri et ses fournisseurs d'images).*
