# MedSuperRes Gen

Super-résolution ×4 (32 → 128 px) de lésions cutanées par SRGAN, et mesure de l'utilité des images
super-résolues pour la classification (projet *IA générative visuelle*, Fiche 07).

## Données

- **DermaMNIST 128 px** (MedMNIST+), 7 classes, partitions officielles train / val / test.
- Entrée basse résolution : sous-échantillonnage bicubique ×4 de l'image 128 px.
- Seconde dégradation (robustesse) : flou gaussien → bicubique ×4 → bruit gaussien, graine fixe.
  Paramètres exacts dans `splits/degradation.json`.
- Les images sources ne sont pas redistribuées dans ce dépôt (`data/` est ignoré par git) ;
  `src/prepare_data.py` les télécharge et les reconstruit à l'identique.

## Démarrage

```bash
pip install -r requirements.txt
python src/prepare_data.py --out data
```

Le script produit :

| Fichier | Contenu |
|---|---|
| `data/processed/{train,val,test}.npz` | `hr`, `lr_bicubic`, `lr_degraded`, `labels` |
| `splits/{train,val,test}.csv` | liste figée : index, étiquette, empreinte SHA-1 de l'image HR |
| `splits/class_counts.csv` | effectifs par classe et par partition (fiche de données) |
| `splits/degradation.json` | paramètres des dégradations |

Les fichiers de `splits/` sont versionnés : ils prouvent que les partitions n'ont pas changé entre les
expériences. Le test n'est jamais utilisé pour entraîner, régler ou sélectionner un modèle.

## Plan des expériences

| Code | Modèle |
|---|---|
| A0 | SRResNet, perte L1 seule |
| A | SRGAN de référence (pixel + adversariale) |
| B | A + blocs d'attention résiduelle |
| C | B + perte de préservation des structures (contours + SSIM/L1) |

Évaluation applicative : un classifieur fixe (ResNet-18) entraîné sur les images HR du train, gelé,
appliqué aux versions bicubique, A0, A, B, C et réelle du test.

## Code repris et assistance IA

- Code de départ autorisé : https://github.com/leftthomas/SRGAN (à citer dans les modules concernés).
- Assistance par outil d'IA générative : Claude (Anthropic) a aidé à rédiger `src/prepare_data.py`
  et ce README. Toute autre assistance sera déclarée ici.
