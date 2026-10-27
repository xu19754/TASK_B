"""Render engineering figures from the saved numerical screening results."""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
for path in ['/System/Library/Fonts/Supplemental/Arial Unicode.ttf', '/System/Library/Fonts/STHeiti Light.ttc']:
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
        break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11, 'figure.facecolor': 'white'})


def read_obj(path):
    v, f = [], []
    for line in path.read_text().splitlines():
        values = line.split()
        if values and values[0] == 'v':
            v.append(list(map(float, values[1:4])))
        elif values and values[0] == 'f':
            f.append([int(x.split('/')[0]) - 1 for x in values[1:]])
    return np.array(v), np.array(f)


def cross_section():
    angles = np.linspace(math.pi, math.pi * 1.5, 13)
    outer = np.vstack([np.c_[-.04 + .100 * np.cos(angles), .10 + .100 * np.sin(angles)], [.14, 0], [.14, .1]])
    inner = np.vstack([np.c_[-.04 + .098 * np.cos(angles), .10 + .098 * np.sin(angles)], [.138, .002], [.138, .1]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, deg in zip(axes, [0, 20]):
        theta = math.radians(deg)
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        shell = np.vstack([outer, inner[::-1]]) @ rot.T * 100
        ax.add_patch(Polygon(shell, closed=True, facecolor='#2e7aaf', edgecolor='#205273', linewidth=1.6))
        ax.set_aspect('equal')
        ax.set_xlim(-23, 19)
        ax.set_ylim(-6, 19)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('前后方向（cm，左侧为狗尾）')
        ax.set_ylabel('相对高度（cm）')
        ax.grid(alpha=.14)
        rim = np.array([-.138, .10]) @ rot.T * 100
        ax.annotate('后侧出口', rim, xytext=(-21, 16), arrowprops={'arrowstyle': '->', 'color': '#b74135'}, color='#b74135')
        ax.annotate('重力', xy=(15, -4), xytext=(15, 2), ha='center', arrowprops={'arrowstyle': '-|>', 'lw': 2}, color='#374151')
        if deg == 0:
            ax.annotate('', xy=(-4, -2), xytext=(13.8, -2), arrowprops={'arrowstyle': '<->'})
            ax.text(4.9, -4, '平底仅 17.8 cm', ha='center')
            ax.annotate('', xy=(-18, 0), xytext=(-18, 10), arrowprops={'arrowstyle': '<->', 'color': '#b74135'})
            ax.text(-19.5, 5, '10 cm', rotation=90, va='center', color='#b74135')
            ax.set_title('当前截面：后沿与其他篮沿同高', fontsize=13, pad=12)
        else:
            transformed = inner @ rot.T * 100
            minimum = transformed[np.argmin(transformed[:, 1])]
            ax.scatter(*minimum, s=75, c='#e49d27', zorder=5)
            ax.annotate('理想小物体仍可停在弧底', minimum, xytext=(-1, 11), fontsize=10,
                        arrowprops={'arrowstyle': '->', 'color': '#7a5c28'}, color='#7a5c28')
            ax.hlines([rim[1], minimum[1]], -20, rim[0], colors='#b74135', linestyles='dashed', linewidth=1)
            ax.annotate('', xy=(-19, minimum[1]), xytext=(-19, rim[1]), arrowprops={'arrowstyle': '<->', 'color': '#b74135'})
            ax.text(-20.5, (rim[1] + minimum[1]) / 2, f'约 {rim[1] - minimum[1]:.1f} cm', rotation=90,
                    va='center', color='#b74135')
            ax.set_title('后倾 20°：出口仍高于重力低点', fontsize=13, pad=12)
    fig.suptitle('篮子截面与倾倒障碍', fontsize=17, y=.98)
    fig.text(.5, .015, '截面几何示意；小物体势垒模型不等同于真实盒子、瓶子或整机动力学。', ha='center', fontsize=9, color='#5c6470')
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(OUT / 'basket_section.png', dpi=170)
    plt.close(fig)


def packing(data):
    trial = next(x for x in data['mixed_trials'] if x['mu'] == .5 and x['tilt_deg'] == 20)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    v, f = read_obj(ROOT / 'assets/b2z1/meshes/basket/basket_visual.obj')
    ax.add_collection3d(Poly3DCollection(v[f] * 100, facecolors='#5e9bb8', alpha=.20, edgecolors=(.1, .3, .4, .15), linewidths=.2))
    palette = ['#ed8d36', '#32a8bd', '#e4c436']
    names = ['糖盒', '芥末瓶', '香蕉']
    cargo_faces, cargo_colors = [], []
    for pose, color, label in zip(trial['initial_body_poses'], palette, names):
        v, f = read_obj(OUT / 'screen_meshes' / (pose['asset'] + '.obj'))
        q = pose['quat_wxyz']
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        points = (v @ r.T + np.array(pose['position_m'])) * 100
        cargo_faces.append(points[f])
        cargo_colors.extend([color] * len(f))
        ax.plot([], [], [], color=color, linewidth=7, label=label)
    ax.add_collection3d(Poly3DCollection(np.concatenate(cargo_faces), facecolors=cargo_colors,
                                       edgecolors=cargo_colors, linewidths=0, shade=True))
    ax.set(xlim=(-15, 15), ylim=(-13, 13), zlim=(0, 12), xlabel='X（cm，负向为狗尾）', ylabel='Y（cm）', zlabel='Z（cm）')
    ax.set_box_aspect((30, 26, 12))
    ax.view_init(elev=32, azim=-125)
    ax.legend(loc='upper left', frameon=False)
    ax.set_title('三件混合物体：指定摆放后的静止初筛', fontsize=15, pad=20)
    fig.text(.5, .035, '按实际网格及 MuJoCo 静止姿态绘制；不是机械臂自动装载或行走运输验证。', ha='center', fontsize=9, color='#5c6470')
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.savefig(OUT / 'packing_three.png', dpi=170)
    plt.close(fig)


def contact_table(data):
    angles = [15, 20, 30, 45, 60, 75]
    friction = [.2, .5, 1.0]
    grid = np.array([[next(r['rear_exit_count'] for r in data['mixed_trials'] if r['mu'] == mu and r['tilt_deg'] == angle)
                      for angle in angles] for mu in friction])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=3, aspect='auto')
    ax.set_xticks(range(6), [f'{a}°' for a in angles])
    ax.set_yticks(range(3), [f'μ = {mu:.1f}' for mu in friction])
    for (i, j), value in np.ndenumerate(grid):
        ax.text(j, i, f'{value}/3', ha='center', va='center', color='white' if value == 0 else '#202020', fontsize=15, fontweight='bold')
    ax.set_xlabel('缓慢后倾角度')
    ax.set_title('从尾部完全排出的数量：三件混合物体', fontsize=15, pad=15)
    fig.text(.5, .02, '独立 MuJoCo 初筛，每格仅一个指定摆放案例；不是 Isaac Sim，也不是成功率统计。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT / 'tilt_results.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    data = json.loads((OUT / 'contact_screen.json').read_text())
    cross_section()
    packing(data)
    contact_table(data)
    print('Figures saved to', OUT)
