import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time
from functools import lru_cache


class Charge:
    def __init__(self, x, y, q):
        self.x = float(x)
        self.y = float(y)
        self.q = float(q)

    def copy(self):
        return Charge(self.x, self.y, self.q)


class FieldLineSimulator:
    # Configuration constants for field line simulation
    CURVATURE_THRESHOLD = np.cos(np.radians(25))  # Detect ~25 degree turns
    SEEDING_RADIUS_SCALE = 0.15  # Base radius for seeding points
    SEEDING_CHARGE_OFFSET = 0.5  # Offset for charge magnitude in radius calculation
    MIN_LINE_LENGTH = 2  # Minimum points for a valid field line
    FIELD_MAGNITUDE_MIN = 1e-8  # Threshold for equilibrium point detection
    
    def __init__(self):
        # default charges
        self.charges = [Charge(0.0, 0.6, -1.0), Charge(-0.6, -0.3, +1.0), Charge(0.6, -0.3, -1.0)]

    # Electric field at (x,y) — accepts numpy arrays or scalars (optimized)
    def electric_field(self, x, y):
        Ex = np.zeros_like(x, dtype=float)
        Ey = np.zeros_like(y, dtype=float)
        for c in self.charges:
            dx = x - c.x
            dy = y - c.y
            r2 = dx * dx + dy * dy
            r2 = np.maximum(r2, 1e-20)  # faster than boolean indexing
            r3 = r2 * np.sqrt(r2)  # avoid r**3 which is slower
            factor = c.q / r3
            Ex += factor * dx
            Ey += factor * dy
        return Ex, Ey

    # single-point field (scalars) - optimized
    def E_at(self, x, y):
        Ex, Ey = 0.0, 0.0
        for c in self.charges:
            dx = x - c.x
            dy = y - c.y
            r2 = dx*dx + dy*dy
            if r2 < 1e-20:
                continue
            r3 = r2 * np.sqrt(r2)  # faster than r**3
            factor = c.q / r3
            Ex += factor * dx
            Ey += factor * dy
        return np.array([Ex, Ey])

    # RK4 step; dir=+1 follow E, dir=-1 follow -E - true RK4 integration
    def rk4_step(self, x, y, h, dir=1):
        # Proper RK4 integration that respects field magnitude
        k1 = dir * self.E_at(x, y)
        mag1 = np.sqrt(k1[0]*k1[0] + k1[1]*k1[1])
        if mag1 < 1e-10:
            return x, y, mag1  # no field, don't move
        k1_norm = k1 / mag1
        
        k2 = dir * self.E_at(x + 0.5*h*k1_norm[0], y + 0.5*h*k1_norm[1])
        mag2 = np.sqrt(k2[0]*k2[0] + k2[1]*k2[1])
        if mag2 < 1e-10:
            return x, y, mag1
        k2_norm = k2 / mag2
        
        k3 = dir * self.E_at(x + 0.5*h*k2_norm[0], y + 0.5*h*k2_norm[1])
        mag3 = np.sqrt(k3[0]*k3[0] + k3[1]*k3[1])
        if mag3 < 1e-10:
            return x, y, mag1
        k3_norm = k3 / mag3
        
        k4 = dir * self.E_at(x + h*k3_norm[0], y + h*k3_norm[1])
        mag4 = np.sqrt(k4[0]*k4[0] + k4[1]*k4[1])
        if mag4 < 1e-10:
            return x, y, mag1
        k4_norm = k4 / mag4
        
        # Weighted average for direction
        E_avg = (k1_norm + 2*k2_norm + 2*k3_norm + k4_norm) / 6.0
        E_avg_mag = np.sqrt(E_avg[0]*E_avg[0] + E_avg[1]*E_avg[1])
        
        # Average magnitude for adaptive stepping
        mag_avg = (mag1 + 2*mag2 + 2*mag3 + mag4) / 6.0
        
        if E_avg_mag < 1e-10:
            return x, y, mag_avg
        E_avg_norm = E_avg / E_avg_mag
        
        return x + h*E_avg_norm[0], y + h*E_avg_norm[1], mag_avg

    # Trace a single line starting from (x0,y0). dir=+1 forward along E, dir=-1 backward.
    def trace_line(self, x0, y0, dir=1, h=0.03, max_steps=2000, stop_radius=0.08):
        # Pre-allocate arrays for speed
        xs = np.zeros(max_steps + 1)
        ys = np.zeros(max_steps + 1)
        xs[0], ys[0] = x0, y0
        x, y = x0, y0
        prev_x, prev_y = x0, y0
        
        for i in range(max_steps):
            # Adaptive step sizing based on multiple factors:
            # 1. Proximity to charges (smaller steps near singularities)
            min_dist_sq = min((x - c.x)**2 + (y - c.y)**2 for c in self.charges)
            min_dist = np.sqrt(min_dist_sq)
            dist_factor = max(0.1, min(1.0, min_dist / 0.2))
            
            # 2. Field magnitude (smaller steps in weak fields for better accuracy)
            # Get current field magnitude
            E = dir * self.E_at(x, y)
            field_mag = np.sqrt(E[0]*E[0] + E[1]*E[1])
            mag_factor = max(0.3, min(2.0, 1.0 / (field_mag + 0.1)))
            
            # 3. Curvature (detect sharp turns)
            if i > 0:
                dx_old = x - prev_x
                dy_old = y - prev_y
                len_old = np.sqrt(dx_old*dx_old + dy_old*dy_old)
                if len_old > 1e-10 and field_mag > 1e-10:
                    # Predict next position direction
                    E_norm = E / field_mag
                    # Compare with previous direction
                    cos_angle = (dx_old*E_norm[0] + dy_old*E_norm[1]) / len_old
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    # Reduce step size if angle changes significantly
                    if cos_angle < self.CURVATURE_THRESHOLD:
                        mag_factor *= 0.5
            
            # Combined adaptive step
            h_loc = h * dist_factor * mag_factor
            
            # Apply RK4 step with adaptive size (single call per iteration)
            prev_x, prev_y = x, y
            x, y, _ = self.rk4_step(x, y, h_loc, dir=dir)
            xs[i+1], ys[i+1] = x, y
            
            # Stop if field is too weak (equilibrium point)
            if field_mag < self.FIELD_MAGNITUDE_MIN:
                return xs[:i+2], ys[:i+2], None
            
            # stop if we hit a charge
            for c in self.charges:
                if (x - c.x)**2 + (y - c.y)**2 < stop_radius**2:
                    return xs[:i+2], ys[:i+2], c
            
            # stop if out of bounds
            if x*x + y*y > 25:
                return xs[:i+2], ys[:i+2], None
        
        return xs, ys, None

    # Build all field lines with seeds near positive charges (forward) and near negative charges (backward)
    def build_field_lines(self, n_per_charge=24, h=0.03, max_steps=2000, stop_radius=0.08):
        lines = []  # each item: (xs, ys)
        
        # Adaptive seeding based on charge magnitude
        for c in self.charges:
            # More lines for larger charges
            n_lines = max(8, int(n_per_charge * np.sqrt(abs(c.q))))
            angles = np.linspace(0, 2*np.pi, n_lines, endpoint=False)
            
            # Seeding distance proportional to charge magnitude (closer for weaker charges)
            r = self.SEEDING_RADIUS_SCALE / np.sqrt(abs(c.q) + self.SEEDING_CHARGE_OFFSET)
            
            if c.q > 0:
                # Positive charges -> integrate forward
                for angle in angles:
                    x0 = c.x + r*np.cos(angle)
                    y0 = c.y + r*np.sin(angle)
                    xs, ys, hit = self.trace_line(x0, y0, dir=1, h=h, max_steps=max_steps, stop_radius=stop_radius)
                    # Keep ALL lines - don't filter by destination
                    if len(xs) > self.MIN_LINE_LENGTH:  # Only keep if line actually went somewhere
                        lines.append((xs, ys))
            else:
                # Negative charges -> integrate backward (follow -E) to find origins
                for angle in angles:
                    x0 = c.x + r*np.cos(angle)
                    y0 = c.y + r*np.sin(angle)
                    xs, ys, hit = self.trace_line(x0, y0, dir=-1, h=h, max_steps=max_steps, stop_radius=stop_radius)
                    # Keep ALL lines - don't filter by origin
                    if len(xs) > self.MIN_LINE_LENGTH:
                        # reverse to have arrow flow from + to - when plotting
                        lines.append((xs[::-1], ys[::-1]))
        
        return lines


class FieldLineGUI:
    def __init__(self, root):
        self.root = root
        root.title("Veldlijnen simulator — sleep, voeg toe, verwijder, exporteer")
        self.sim = FieldLineSimulator()

        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Performance optimization
        self.last_drag_update = 0
        self.drag_update_interval = 0.05  # 50ms between updates during drag
        self.cached_lines = None

        self.create_widgets()
        self.selected_idx = None
        self.scatter_artists = []
        self.lines_artists = []
        self.dragging = False
        self.drag_idx = None

        self.redraw()

    def create_widgets(self):
        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(frm, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text="Ladingen").pack()
        self.listbox = tk.Listbox(left, width=34, height=12)
        self.listbox.pack(pady=4)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Voeg toe", command=self.add_charge).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="Verwijder", command=self.remove_charge).pack(side=tk.LEFT, fill=tk.X, expand=True)

        editf = ttk.Frame(left)
        editf.pack(fill=tk.X, pady=6)
        ttk.Label(editf, text="Q:").grid(row=0, column=0)
        self.q_var = tk.DoubleVar()
        self.q_spin = ttk.Spinbox(editf, textvariable=self.q_var, from_=-10, to=10, increment=0.1, width=10, command=self.on_q_change)
        self.q_spin.grid(row=0, column=1, padx=4)
        ttk.Label(editf, text="x:").grid(row=1, column=0)
        self.x_var = tk.DoubleVar()
        self.x_entry = ttk.Entry(editf, textvariable=self.x_var, width=12)
        self.x_entry.grid(row=1, column=1, padx=4)
        ttk.Label(editf, text="y:").grid(row=2, column=0)
        self.y_var = tk.DoubleVar()
        self.y_entry = ttk.Entry(editf, textvariable=self.y_var, width=12)
        self.y_entry.grid(row=2, column=1, padx=4)
        ttk.Button(editf, text="Toepassen", command=self.apply_position).grid(row=3, column=0, columnspan=2, pady=6)

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        opts = ttk.Frame(left)
        opts.pack(fill=tk.X)
        ttk.Label(opts, text="Opties:").pack(anchor=tk.W)
        
        # Quality preset
        ttk.Label(opts, text="Kwaliteit:").pack(anchor=tk.W)
        self.quality_var = tk.StringVar(value="Laag (snelst)")
        quality_combo = ttk.Combobox(opts, textvariable=self.quality_var, 
                                     values=["Laag (snelst)", "Gemiddeld", "Hoog"], 
                                     state="readonly", width=18)
        quality_combo.pack(anchor=tk.W, pady=2)
        quality_combo.bind('<<ComboboxSelected>>', self.on_quality_change)
        
        self.density_var = tk.IntVar(value=16)
        ttk.Label(opts, text="lijnen per lading:").pack(anchor=tk.W, pady=(6,0))
        ttk.Scale(opts, from_=6, to=48, variable=self.density_var, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Button(left, text="Plot veldlijnen", command=self.redraw).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Exporteer PNG", command=self.export_png).pack(fill=tk.X)

        undo_frame = ttk.Frame(left)
        undo_frame.pack(fill=tk.X, pady=4)
        ttk.Button(undo_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(undo_frame, text="Redo", command=self.redo).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Right: matplotlib figure
        right = ttk.Frame(frm)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def push_undo(self):
        self.undo_stack.append([c.copy() for c in self.sim.charges])
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append([c.copy() for c in self.sim.charges])
        self.sim.charges = self.undo_stack.pop()
        self.redraw()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append([c.copy() for c in self.sim.charges])
        self.sim.charges = self.redo_stack.pop()
        self.redraw()

    def add_charge(self):
        self.push_undo()
        # ask for q
        q = simpledialog.askfloat("Voeg lading toe", "q (positief/negatief):", initialvalue=1.0, parent=self.root)
        if q is None:
            return
        # default position at center
        c = Charge(0.0, 0.0, q)
        self.sim.charges.append(c)
        self.redraw()

    def remove_charge(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.push_undo()
        idx = sel[0]
        del self.sim.charges[idx]
        self.redraw()

    def on_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.selected_idx = idx
        c = self.sim.charges[idx]
        self.q_var.set(c.q)
        self.x_var.set(round(c.x, 3))
        self.y_var.set(round(c.y, 3))

    def on_q_change(self):
        if self.selected_idx is None:
            return
        try:
            q = float(self.q_var.get())
        except Exception:
            return
        self.push_undo()
        self.sim.charges[self.selected_idx].q = q
        self.redraw()

    def apply_position(self):
        if self.selected_idx is None:
            return
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
        except Exception:
            return
        self.push_undo()
        self.sim.charges[self.selected_idx].x = x
        self.sim.charges[self.selected_idx].y = y
        self.redraw()

    def on_quality_change(self, event=None):
        quality = self.quality_var.get()
        if quality == "Laag (snelst)":
            self.density_var.set(10)
        elif quality == "Gemiddeld":
            self.density_var.set(16)
        else:  # Hoog
            self.density_var.set(32)
        self.redraw()
    
    def export_png(self):
        fname = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG image','*.png')])
        if not fname:
            return
        self.fig.savefig(fname, dpi=300)
        messagebox.showinfo('Export', f'Afbeelding opgeslagen als {fname}')

    def redraw(self):
        # update listbox
        self.listbox.delete(0, tk.END)
        for c in self.sim.charges:
            self.listbox.insert(tk.END, f"q={c.q:.2f}  at ({c.x:.2f},{c.y:.2f})")
        # compute field lines with optimized parameters
        n = int(self.density_var.get())
        lines = self.sim.build_field_lines(n_per_charge=n, h=0.04, max_steps=1500, stop_radius=0.08)
        self.cached_lines = lines  # cache for potential reuse

        self.ax.clear()
        # stream-like drawing: draw lines in orange with reduced linewidth for speed
        for xs, ys in lines:
            self.ax.plot(xs, ys, color='darkorange', linewidth=0.7, antialiased=True)

        # draw charges as big circles with +/−
        self.scatter_artists.clear()
        for i, c in enumerate(self.sim.charges):
            color = 'red' if c.q > 0 else 'deepskyblue'
            s = 200  # slightly smaller for faster rendering
            art = self.ax.scatter(c.x, c.y, s=s, color=color, zorder=3, picker=True, edgecolors='none')
            self.scatter_artists.append(art)
            label = '+' if c.q > 0 else '−'
            self.ax.text(c.x, c.y, label, color='white', ha='center', va='center', 
                        fontsize=13, fontweight='bold', zorder=4)

        self.ax.set_xlim(-1.6, 1.6)
        self.ax.set_ylim(-1.6, 1.6)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.canvas.draw_idle()  # use draw_idle for better performance

    # Mouse interaction for dragging charges
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # check if clicked near a charge
        for i, c in enumerate(self.sim.charges):
            if np.hypot(event.xdata - c.x, event.ydata - c.y) < 0.12:
                self.dragging = True
                self.drag_idx = i
                self.push_undo()
                return

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax or self.drag_idx is None:
            return
        
        # Throttle updates for performance
        current_time = time.time()
        if current_time - self.last_drag_update < self.drag_update_interval:
            return
        self.last_drag_update = current_time
        
        # move charge to cursor
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self.sim.charges[self.drag_idx].x = x
        self.sim.charges[self.drag_idx].y = y
        
        # update entries if selected
        if self.selected_idx == self.drag_idx:
            self.x_var.set(round(x, 3)); self.y_var.set(round(y, 3))
        
        # Ultra-fast preview: minimal lines, larger steps
        self.ax.clear()
        lines = self.sim.build_field_lines(n_per_charge=6, h=0.08, max_steps=400, stop_radius=0.1)
        for xs, ys in lines:
            self.ax.plot(xs, ys, color='darkorange', linewidth=0.6, alpha=0.7)
        
        for i, c in enumerate(self.sim.charges):
            color = 'red' if c.q > 0 else 'deepskyblue'
            self.ax.scatter(c.x, c.y, s=180, color=color, zorder=3, edgecolors='none')
            label = '+' if c.q > 0 else '−'
            self.ax.text(c.x, c.y, label, color='white', ha='center', va='center', 
                        fontsize=12, fontweight='bold', zorder=4)
        
        self.ax.set_xlim(-1.6, 1.6)
        self.ax.set_ylim(-1.6, 1.6)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.canvas.draw_idle()  # use draw_idle for async rendering

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.drag_idx = None
            # full redraw with requested density
            self.redraw()


if __name__ == '__main__':
    root = tk.Tk()
    app = FieldLineGUI(root)
    root.geometry('1100x700')
    root.mainloop()