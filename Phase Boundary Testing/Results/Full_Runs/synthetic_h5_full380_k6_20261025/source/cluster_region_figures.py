"""Discrete cluster-region figures. No smoothing or semantic relabeling."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

COLORS = ['#00a4ff', '#ff334b', '#d7ef00', '#bd59ff', '#00df95', '#ff981f']


def palette(k):
    colors = COLORS[:k] if k <= len(COLORS) else plt.get_cmap('tab20')(np.linspace(0,1,k))
    cmap = ListedColormap(colors)
    cmap.set_bad((0,0,0,0))
    return colors, cmap, BoundaryNorm(np.arange(k+1)-.5, k)


def nearest_sample_regions(positions, predictions, shape):
    positions, predictions = np.asarray(positions), np.asarray(predictions)
    if positions.shape != (len(predictions),2) or not np.isfinite(positions).all():
        raise ValueError('One finite x/y coordinate is required per assignment')
    if len(np.unique(positions,axis=0)) != len(positions):
        raise ValueError('Duplicate coordinates cannot define unique regions')
    rows,cols=np.indices(shape)
    distance,index=cKDTree(positions).query(np.column_stack([cols.ravel(),rows.ravel()]))
    return predictions[index].reshape(shape), index.reshape(shape), distance.reshape(shape)


def save_region_figure(background, grid, k, output, title, positions=None, predictions=None,
                       interpolation_note='Saved assignments; no interpolation'):
    output=Path(output)
    colors,cmap,norm=palette(k)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    axes[0].imshow(background,cmap='gray',origin='upper')
    axes[0].imshow(grid,cmap=cmap,norm=norm,alpha=.33,origin='upper',interpolation='nearest')
    if positions is not None:
        axes[0].scatter(positions[:,0],positions[:,1],c=predictions,cmap=cmap,norm=norm,
                        s=17,linewidths=.15,edgecolors='white')
    axes[0].set_title('Cluster regions on original image')
    axes[1].imshow(grid,cmap=cmap,norm=norm,origin='upper',interpolation='nearest')
    axes[1].set_title('Cluster regions')
    for ax in axes:
        ax.set(xlabel='x / column (pixels)',ylabel='y / row (pixels)')
    handles=[Patch(facecolor=colors[c],label=f'Cluster {c}'+(' (unoccupied)' if not np.any(grid==c) else '')) for c in range(k)]
    fig.legend(handles=handles,loc='lower center',ncol=min(k,6),frameon=False,bbox_to_anchor=(.5,.03))
    fig.suptitle(title,fontsize=15)
    fig.text(.5,.005,interpolation_note,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(output,dpi=180)
    plt.close(fig)


def save_atomic_figure(background, positions, predictions, k, output, title):
    colors,cmap,norm=palette(k)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(background,cmap='gray',origin='upper')
    ax.scatter(positions[:,0],positions[:,1],c=predictions,cmap=cmap,norm=norm,s=21,
               linewidths=.15,edgecolors='white')
    ax.set(title=title,xlabel='Original x (pixels)',ylabel='Original y (pixels)')
    fig.legend(handles=[Patch(facecolor=colors[c],label=f'Cluster {c}') for c in range(k)],
               loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(output,dpi=180)
    plt.close(fig)
