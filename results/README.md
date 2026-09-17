# Index des résultats

| Chemin | État et usage |
|---|---|
| `roll_gain_comparison/` | Série complète : huit essais, deux réglages, deux signes et deux tirages ; traces, protocole, environnement et bilan |
| `readout_v2/` | Calibration et comparaison hors boucle ; ensembles train/validation/test et coefficients. Ce n'est pas une validation de vol |
| `roll_closed_loop_pilot/` | Premier pilote, dont un échec de normalisation ; résumé conservé, sans trajectoires complètes |
| `roll_closed_loop_pilot_robust/` | Pilote avec plancher de normalisation ; décodeur utilisé dans la comparaison actuelle |
| `roll_paired_confirmation/` | Série interrompue et partiellement sauvegardée. Six traces présentes sur les 32 essais prévus, pas de conclusion globale |
| `paired_neural_probe_20260916.json` | Ancien diagnostic sensorimoteur hors boucle |

Les échecs restent disponibles. Ne pas fusionner ces séries en un taux de réussite :
protocoles, objectifs et niveaux de complétude diffèrent. Les fichiers de calibration
et de décodeur sont nécessaires à la reproduction et ne doivent pas être déplacés.
