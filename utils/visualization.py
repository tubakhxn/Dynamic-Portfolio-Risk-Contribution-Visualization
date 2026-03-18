import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgba
import matplotlib.patheffects as patheffects

# --- Color and style helpers ---
def get_asset_colors(n_assets):
    # Use a perceptually uniform colormap
    base_cmap = cm.get_cmap('plasma', n_assets)
    return [base_cmap(i) for i in range(n_assets)]

def dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#111111',
        'figure.facecolor': '#000000',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': '#dddddd',
        'xtick.color': '#bbbbbb',
        'ytick.color': '#bbbbbb',
        'grid.color': '#333333',
        'legend.frameon': False,
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'legend.fontsize': 12,
        'font.size': 12,
    })

def animate_stacked_risk_contributions(
    rc, sigma_p, asset_names, window=100, fps=40, highlight=True
):
    """
    Animate stacked area risk contributions.
    rc: (n_steps, n_assets)
    sigma_p: (n_steps,)
    asset_names: list of str
    """
    import matplotlib.animation as animation
    n_steps, n_assets = rc.shape
    colors = get_asset_colors(n_assets)
    dark_theme()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_steps)
    # For fading old data
    fade_alpha = np.linspace(0.2, 1.0, window)
    # For glow: draw multiple alpha layers
    def plot_frame(i):
        ax.clear()
        ax.set_facecolor('#111111')
        start = max(0, i-window+1)
        end = i+1
        rc_win = rc[start:end]
        sigma_win = sigma_p[start:end]
        x_win = x[start:end]
        y_stack = np.cumsum(rc_win, axis=1)
        y_base = np.zeros(len(x_win))
        for j in range(n_assets):
            y1 = y_base.copy()
            y2 = y_stack[:, j]
            # Draw fading and glow per time step
            for t in range(len(x_win)-1):
                seg_x = x_win[t:t+2]
                seg_y1 = y1[t:t+2]
                seg_y2 = y2[t:t+2]
                seg_alpha = fade_alpha[-len(x_win)+t] * 0.7 + 0.3
                # Glow: blurred underlayers
                for glow in range(6, 0, -1):
                    ax.fill_between(
                        seg_x, seg_y1, seg_y2,
                        color=to_rgba(colors[j], alpha=0.08*glow*seg_alpha),
                        linewidth=0
                    )
                # Main area
                ax.fill_between(
                    seg_x, seg_y1, seg_y2,
                    color=to_rgba(colors[j], alpha=seg_alpha),
                    linewidth=0
                )
            # Glowing edge
            ax.plot(
                x_win, y2,
                color=to_rgba(colors[j], alpha=0.9),
                linewidth=2.5, solid_capstyle='round',
                path_effects=[
                    patheffects.withStroke(linewidth=6, foreground=to_rgba(colors[j], alpha=0.25))
                ]
            )
            y_base = y2
        # Portfolio volatility line
        ax.plot(x_win, sigma_win, color='#ffffff', lw=2, alpha=0.7, label='Portfolio Volatility')
        # Highlight dominant asset
        if highlight:
            dom_idx = np.argmax(rc[i])
            ax.plot(
                [x[i]], [y_stack[-1, dom_idx]],
                marker='o', markersize=14, markeredgewidth=2,
                markerfacecolor=to_rgba(colors[dom_idx], alpha=0.7),
                markeredgecolor='#fff',
                zorder=10
            )
        ax.set_xlim(x_win[0], x_win[-1])
        ax.set_ylim(0, np.max(sigma_p)*1.15)
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Dynamic Risk Contribution Breakdown')
        ax.grid(True, alpha=0.25)
        # Legend
        handles = [plt.Line2D([0], [0], color=colors[j], lw=4, label=asset_names[j]) for j in range(n_assets)]
        handles.append(plt.Line2D([0], [0], color='#fff', lw=2, label='Portfolio Volatility'))
        ax.legend(handles=handles, loc='upper left', frameon=False, ncol=2)
    # Interpolate between frames for ultra-smoothness
    n_interp = 2  # number of interpolated frames between each real frame
    total_frames = (rc.shape[0]-1) * n_interp + 1
    def interp_frame(idx):
        base = idx // n_interp
        frac = (idx % n_interp) / n_interp
        if frac == 0 or base == rc.shape[0]-1:
            plot_frame(base)
        else:
            # Linear interpolation between base and base+1
            rc_interp = rc[base] * (1-frac) + rc[base+1] * frac
            sigma_interp = sigma_p[base] * (1-frac) + sigma_p[base+1] * frac
            rc_stack = np.vstack([rc[:base], rc_interp])
            sigma_stack = np.concatenate([sigma_p[:base], [sigma_interp]])
            plot_frame_interp(rc_stack, sigma_stack, base)

    def plot_frame_interp(rc_stack, sigma_stack, i):
        # Same as plot_frame, but for interpolated stack
        ax.clear()
        ax.set_facecolor('#111111')
        start = max(0, i-window+1)
        end = i+1
        rc_win = rc_stack[start:end]
        sigma_win = sigma_stack[start:end]
        x_win = np.arange(start, end)
        y_stack = np.cumsum(rc_win, axis=1)
        y_base = np.zeros(len(x_win))
        for j in range(rc_win.shape[1]):
            y1 = y_base.copy()
            y2 = y_stack[:, j]
            for t in range(len(x_win)-1):
                seg_x = x_win[t:t+2]
                seg_y1 = y1[t:t+2]
                seg_y2 = y2[t:t+2]
                seg_alpha = fade_alpha[-len(x_win)+t] * 0.7 + 0.3
                for glow in range(6, 0, -1):
                    ax.fill_between(
                        seg_x, seg_y1, seg_y2,
                        color=to_rgba(colors[j], alpha=0.08*glow*seg_alpha),
                        linewidth=0
                    )
                ax.fill_between(
                    seg_x, seg_y1, seg_y2,
                    color=to_rgba(colors[j], alpha=seg_alpha),
                    linewidth=0
                )
            ax.plot(
                x_win, y2,
                color=to_rgba(colors[j], alpha=0.9),
                linewidth=2.5, solid_capstyle='round',
                path_effects=[
                    patheffects.withStroke(linewidth=6, foreground=to_rgba(colors[j], alpha=0.25))
                ]
            )
            y_base = y2
        ax.plot(x_win, sigma_win, color='#ffffff', lw=2, alpha=0.7, label='Portfolio Volatility')
        if highlight:
            dom_idx = np.argmax(rc_stack[-1])
            ax.plot(
                [x_win[-1]], [y_stack[-1, dom_idx]],
                marker='o', markersize=14, markeredgewidth=2,
                markerfacecolor=to_rgba(colors[dom_idx], alpha=0.7),
                markeredgecolor='#fff',
                zorder=10
            )
        ax.set_xlim(x_win[0], x_win[-1])
        ax.set_ylim(0, np.max(sigma_p)*1.15)
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Dynamic Risk Contribution Breakdown')
        ax.grid(True, alpha=0.25)
        handles = [plt.Line2D([0], [0], color=colors[j], lw=4, label=asset_names[j]) for j in range(rc_win.shape[1])]
        handles.append(plt.Line2D([0], [0], color='#fff', lw=2, label='Portfolio Volatility'))
        ax.legend(handles=handles, loc='upper left', frameon=False, ncol=2)

    ani = animation.FuncAnimation(
        fig, interp_frame, frames=total_frames, interval=1000/fps/n_interp, repeat=False
    )
    plt.show()
