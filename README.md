# Yiying Garden walk-along visualization

Code and figures for walking maps in Yiying Garden (怡影园). Paths use a scenery-change score Q plus a crowd-aware cost, then get drawn as winding garden routes.

Zeng Wenquan · https://github.com/chaosbull

## Layout

```
data/       22 spots, 55 roads, scenic_generated.xlsx
results/    Q scores, fitted weights, recommendation tables
src/        generator, model, and the five figure scripts
output/     GIFs, overview maps, CSVs (frame PNGs not shipped)
```

Frame folders under `output/**/frames/` are written when you rebuild and are listed in `.gitignore`. The GIFs and overview maps are already in `output/`.

## Run

```bash
pip install -r requirements.txt
python src/build.py
```

Default is 400 frames. Examples: `python src/build.py --frames 200`, or `--lang zh` for Chinese labels.

Refresh the network tables with `python src/generate_scenic_data.py`.

## What ends up in output/

1. `01_network_scores/` — whole network, Q marked on each segment
2. `02_recommended_paths/` — three gate-to-exit routes
3. `03_frame_walk/` — walk GIF; recommendations only when you arrive at a spot
4. `04_midtrip_reroute/` — change the destination once or twice mid-walk
5. `05_aimless_twostep/` — local two-step options from where you are standing

## License

Apache-2.0. See `LICENSE`.
