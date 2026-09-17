# Guide du code

## Parcours principal

1. `scripts/download_malecns_data.py` télécharge les sources officielles.
2. `scripts/build_malecns_cache_v3.py` prépare les tableaux utilisés par `flybrain/lif.py`.
3. `scripts/roll_gain_comparison.py` compare deux coefficients avec le même décodeur.
4. `scripts/report_roll_gain_comparison.py` résume les traces ; `scripts/render_readme_media.py` les illustre.

## Dépendances conservées à leur emplacement historique

| Module | Pourquoi il reste actif |
|---|---|
| `demo_neuroflight_3d_v2.py` | Contrôleur de base, encodeur, chargement Crazyflie et fonctions de rotation |
| `benchmark_motor10_readout.py` | Classe `Motor10Readout`, utilisée par `RawMotors` |
| `validate_neural_control_loop.py` | Constantes et fonctions importées par le module précédent |
| `compare_readout_v2.py` | `RawMotors`, features et régression du décodeur |
| `roll_closed_loop.py` | Protocole de simulation et commandes de couple |

Leurs noms ne sont pas une recommandation de lancer les anciennes démonstrations
3D. Leur contenu reste inchangé pour préserver les empreintes des essais sauvegardés.
Une extraction future de ces classes vers un package devra versionner le nouveau
protocole et vérifier la reproduction numérique.

## Diagnostics conservés

`check_malecns_status.py`, `check_actuator_direction.py` et
`inspect_encoder_overlap.py` inspectent les données, les commandes et l'encodage.
`test_readout_v2.py` et `test_roll_adapter.py` vérifient les calculs et conventions.
`roll_paired_confirmation.py` conserve le protocole d'une série partiellement
sauvegardée : son existence ne signifie pas que tous ses essais ont été terminés.

## Réorganisation

Les anciens lanceurs de démonstration et recherches ponctuelles sont déplacés
dans `archive/experiments/`, sans modification de leur contenu. Le lanceur racine
`main.py` devient `archive/demos/altitude_only.py` : il contrôlait seulement
l'altitude et ne démontrait pas de commande neuronale.
Les packages vides `env/` et `training/` sont conservés sous
`archive/ppo_prototypes/`. Les résultats existants gardent leur chemin, car les
scripts et les protocoles les référencent.
