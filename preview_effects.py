"""Render a contact sheet and animation without a webcam or seal templates."""
import argparse
from pathlib import Path

import cv2
import numpy as np

import effects as FX


def background(width, height):
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = (32, 27, 24)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (40, 35, 32), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (40, 35, 32), 1)
    return frame


def label(frame, text):
    cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('previews'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    width, height = 480, 360
    layers = FX.GlowLayers((height, width, 3))
    tiles = []
    for title, kind, value in (('Rasengan / forming', 'orb', 0.2),
                                ('Rasengan / charged', 'orb', 1),
                                ('Wind / emerging', 'wind', 0.45),
                                ('Rasenshuriken / held', 'wind', 1),
                                ('Rasenshuriken / thrown', 'throw', 1),
                                ('Wind detonation', 'burst', 1)):
        layers.reset()
        if kind == 'orb':
            FX.draw_rasengan(layers, 240, 190, 32 + value * 16, 35, value)
        elif kind == 'wind':
            FX.draw_rasenshuriken(layers, 240, 190, 32, 35,
                                  emergence=value, energize_frame=14 * value)
        elif kind == 'throw':
            blast = FX.JutsuBlast(80, 230, 1, -0.25, 'rasenshuriken',
                                  radius=28, speed=18, animation_frame=35)
            for _ in range(8):
                blast.update()
            blast.draw(layers)
        else:
            burst = FX.WindDome(240, 190, max_radius=165)
            burst.update(9)
            burst.draw(layers)
        tiles.append(label(layers.composite(background(width, height)), title))
    sheet = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
    if not cv2.imwrite(str(args.output / 'effects.png'), sheet):
        raise RuntimeError('Could not write preview image')

    writer = cv2.VideoWriter(str(args.output / 'effects.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 30, (960, 360))
    if not writer.isOpened():
        raise RuntimeError('OpenCV could not open the MP4 encoder')
    effects = []
    try:
        for tick in range(180):
            layers.reset()
            if tick < 75:
                FX.draw_rasengan(layers, 240, 190, 22 + tick / 75 * 10, tick, tick / 75)
            else:
                FX.draw_rasenshuriken(layers, 240, 190, 32, tick,
                                      emergence=min(1, (tick - 75) / 14), energize_frame=tick - 75)
            held = label(layers.composite(background(width, height)), 'Charge / transform / spin')
            layers.reset()
            if tick in (10, 90):
                style = 'rasengan' if tick == 10 else 'rasenshuriken'
                effects.append(FX.JutsuBlast(80, 190, 1, -0.12, style, speed=20,
                                            bounds=(width, height), animation_frame=tick))
            effects = FX.advance_projectiles(effects, layers, width, height)
            flight = label(layers.composite(background(width, height)), 'Throw / trail / impact')
            writer.write(np.hstack((held, flight)))
    finally:
        writer.release()
    print(f'Preview image: {args.output / "effects.png"}')
    print(f'Preview video: {args.output / "effects.mp4"}')


if __name__ == '__main__':
    main()
