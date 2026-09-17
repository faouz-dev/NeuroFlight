# Voir les essais dans une fenêtre

Depuis la racine du dépôt, avec l'environnement Python activé :

```bash
python -m pip install -r requirements.txt
python -m scripts.replay_roll
```

Une fenêtre MuJoCo rejoue le cas diagnostic +12° avec le réglage plus doux.
Fermer la fenêtre pour arrêter. Aucun téléchargement du connectome n'est nécessaire.

```bash
# Cas qui échoue encore au critère, au ralenti :
python -m scripts.replay_roll --angle -12 --speed 0.5
# Ancien réglage, même tirage :
python -m scripts.replay_roll --angle -12 --variant baseline
# Tirage de confirmation :
python -m scripts.replay_roll --seed 1200001 --angle -12
```

Il s'agit d'une **relecture des angles enregistrés**, pas d'un nouveau calcul ni
d'une vidéo d'un drone réel. La translation est recentrée ; seul le roulis est
reconstruit. Le statut du critère est imprimé dans le terminal.
Sur macOS, le visualiseur passif MuJoCo se lance avec `mjpython` à la place de
`python`. Sur Windows/Linux avec affichage local, utiliser `python`.

## Recalculer avec les données déjà téléchargées

Les trois fichiers Feather doivent être dans `data/male_cns/`. Si le cache
`data/male_cns/cache_v3/` n'existe pas, le construire sans téléchargement :

```bash
python -m scripts.build_malecns_cache_v3
```

Pour de nouveaux calculs, déplacer d'abord `results/roll_gain_comparison` vers un
dossier de sauvegarde : sans cela, les scripts réutilisent les traces présentes.
Puis lancer, successivement :

```bash
python -m scripts.roll_gain_comparison
python -m scripts.roll_gain_comparison --confirmation
python -m scripts.report_roll_gain_comparison
python -m scripts.replay_roll
```

Le calcul neuronal est sans fenêtre et peut durer plusieurs minutes ; la relecture
animée s'ouvre ensuite. Conserver `results/readout_v2/` et
`results/roll_closed_loop_pilot_robust/decoder.json`, nécessaires à ces essais.
