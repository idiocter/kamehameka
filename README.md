# Wizard

Weave **serpent → tiger → horse → hare → boar → ram** in front of your webcam to form a Rasengan. Hold to charge, squeeze to transform it into a Rasenshuriken, then open your hand and swing to throw.

## Run

```sh
python -m pip install -r requirements.txt
python main.py --train  # Record your own seal templates, then press s to save.
python main.py
```

Use `venv/bin/python` in place of `python` if using the repository's existing virtual environment. Press `r` to reset all effects and the sequence; press `q` or Escape to quit. Local templates are stored in ignored `seals.json`.

## Effects and controls

- The blue Rasengan has layered rotating currents, a luminous centre, and a soft halo. Its size follows your hand's distance from the camera and grows smoothly while charging.
- After roughly 2.5 seconds, squeeze for four consecutive detections to release four white wind blades. Their growth eases into place and their rotation accelerates smoothly.
- Open your hand and swing for two consecutive fast detections to throw. Faster swings launch faster projectiles. Reacquiring a lost hand does not count as a throw.
- Thrown effects retain their animation and leave fading wakes. A Rasenshuriken's fuse shortens near the image boundary so it detonates on screen, followed by a fading wind splash.

Animation, charge, and projectile movement use elapsed time at a nominal 30 Hz. Long stalls are capped at 100 ms per update. Gesture debounce still counts camera detections. These are procedural OpenCV effects; this app does not detect physical impact surfaces or depth occlusion.

## Preview without a camera

```sh
python preview_effects.py
# Writes previews/effects.png and previews/effects.mp4 (ignored by Git).
python preview_effects.py --output /tmp/wizard-preview
```

The six-second video shows charging, transformation, spinning, both throws, and a wind explosion. Review it before trying the webcam; rendering tests cannot verify hand tracking or visual alignment on your own hand.

## Checks

```sh
python -m unittest discover -s tests -v
```

The headless tests cover rendering at image edges, spin and throw continuity, frame-rate-independent travel, pixel-correct direction, hand scale, and explosion lifecycle. No lint or typecheck tasks are configured.
