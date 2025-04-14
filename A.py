"""Adsorption Data Analyzer
Provides a GUI to fit adsorption kinetics (pseudo-first-order, pseudo-second-order, Elovich, intraparticle diffusion)
and isotherm models (Langmuir, Freundlich) to experimental data.
Offers visualization of data and fitted curves, and computes optional statistics (RMSE, AIC) and residual plots.
"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

# Use a clean matplotlib style for better visuals
plt.style.use('seaborn-v0_8')  # apply a seaborn style for consistency&#8203;:contentReference[oaicite:11]{index=11}

# Define adsorption model functions
def pfo_func(t, Qe, k1):
    """Pseudo-first-order kinetic model: Qt = Qe * (1 - exp(-k1 * t))"""
    return Qe * (1 - np.exp(-k1 * t))

def pso_func(t, Qe, k2):
    """Pseudo-second-order kinetic model: Qt = (Qe^2 * k2 * t) / (1 + Qe * k2 * t)"""
    return (Qe**2 * k2 * t) / (1 + Qe * k2 * t)

def elovich_func(t, a, b):
    """Elovich kinetic model: Qt = (1/b) * ln(1 + a * b * t)"""
    return (1.0/b) * np.log(1 + a * b * t)

def ipd_func(t, k, C):
    """Intraparticle diffusion kinetic model: Qt = k * sqrt(t) + C"""
    return k * np.sqrt(t) + C

def langmuir_func(Ce, Qmax, KL):
    """Langmuir isotherm: Qe = (Qmax * KL * Ce) / (1 + KL * Ce)"""
    return (Qmax * KL * Ce) / (1 + KL * Ce)

def freundlich_func(Ce, Kf, n):
    """Freundlich isotherm: Qe = Kf * Ce^(1/n)"""
    return Kf * (Ce ** (1.0/n))

class Tooltip:
    """A tooltip that appears on hovering over a widget."""
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.id = None
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)
        widget.bind("<ButtonPress>", self.on_leave)
    def on_enter(self, event=None):
        self.schedule()
    def on_leave(self, event=None):
        self.unschedule()
        self.hide()
    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.delay, self.show)
    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
    def show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        label = ttk.Label(tw, text=self.text, background="#ffffe0", borderwidth=1, relief=tk.SOLID)
        label.pack(ipadx=1, ipady=1)
        tw.wm_geometry(f"+{x}+{y}")
    def hide(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

class AdsorptionApp:
    def __init__(self, root):
        self.root = root
        root.title("Adsorption Data Analyzer")
        self.x_data = None
        self.y_data = None
        self.file_path = None
        # Variables to track UI state
        self.analysis_mode = tk.StringVar(value="kinetic")
        self.kin_models_vars = {
            "Pseudo-First Order": tk.BooleanVar(value=True),
            "Pseudo-Second Order": tk.BooleanVar(value=True),
            "Elovich": tk.BooleanVar(value=False),
            "Intraparticle Diffusion": tk.BooleanVar(value=False)
        }
        self.iso_models_vars = {
            "Langmuir": tk.BooleanVar(value=True),
            "Freundlich": tk.BooleanVar(value=True)
        }
        self.calc_stats = tk.BooleanVar(value=True)
        self.show_residual = tk.BooleanVar(value=False)
        # Build the UI
        self.create_widgets()
    def create_widgets(self):
        # File selection frame
        file_frame = ttk.Frame(self.root)
        ttk.Label(file_frame, text="Data File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_entry = ttk.Entry(file_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.file_entry.configure(state="readonly")
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.load_file)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        file_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.root.grid_columnconfigure(1, weight=1)  # make file_entry expandable

        # Analysis type (Kinetic or Isotherm)
        mode_frame = ttk.Frame(self.root)
        ttk.Label(mode_frame, text="Analysis Type:").grid(row=0, column=0, padx=5, pady=5)
        rb1 = ttk.Radiobutton(mode_frame, text="Kinetic", variable=self.analysis_mode, value="kinetic", command=self.update_mode)
        rb2 = ttk.Radiobutton(mode_frame, text="Isotherm", variable=self.analysis_mode, value="isotherm", command=self.update_mode)
        rb1.grid(row=0, column=1, padx=5, pady=5)
        rb2.grid(row=0, column=2, padx=5, pady=5)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="w")

        # Model selection frames
        self.kin_frame = ttk.LabelFrame(self.root, text="Kinetic Models")
        for i, (model, var) in enumerate(self.kin_models_vars.items()):
            ttk.Checkbutton(self.kin_frame, text=model, variable=var).grid(row=i, column=0, sticky="w", padx=5, pady=2)
        self.kin_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        self.iso_frame = ttk.LabelFrame(self.root, text="Isotherm Models")
        for j, (model, var) in enumerate(self.iso_models_vars.items()):
            ttk.Checkbutton(self.iso_frame, text=model, variable=var).grid(row=j, column=0, sticky="w", padx=5, pady=2)
        self.iso_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        # Show only the relevant model frame initially
        if self.analysis_mode.get() == "kinetic":
            self.iso_frame.grid_remove()
        else:
            self.kin_frame.grid_remove()

        # Advanced options (stats and residuals)
        adv_frame = ttk.Frame(self.root)
        ttk.Checkbutton(adv_frame, text="Calculate RMSE & AIC", variable=self.calc_stats).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(adv_frame, text="Show Residual Plot", variable=self.show_residual).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        adv_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        # Results output area with scrollbar
        result_frame = ttk.LabelFrame(self.root, text="Results")
        self.result_text = tk.Text(result_frame, width=80, height=10)
        self.result_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.result_text.configure(state="disabled")
        result_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.result_text['yscrollcommand'] = result_scroll.set
        result_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.root.grid_rowconfigure(4, weight=1)  # make results frame expandable

        # Plot display area
        plot_frame = ttk.LabelFrame(self.root, text="Plot")
        self.fig = plt.Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plot_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.root.grid_rowconfigure(5, weight=1)  # make plot frame expandable

        # Analyze action button
        action_frame = ttk.Frame(self.root)
        analyze_btn = ttk.Button(action_frame, text="Analyze Data", command=self.analyze_data)
        analyze_btn.grid(row=0, column=0, padx=5, pady=5)
        action_frame.grid(row=6, column=0, columnspan=3, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_label.grid(row=7, column=0, columnspan=3, sticky="we")

        # Menu bar with Help and About
        menubar = tk.Menu(self.root)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="User Guide", command=self.show_help)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)

        # Tooltips for key widgets
        Tooltip(browse_btn, "Open a data file (CSV or TXT) with two columns of data")
        Tooltip(rb1, "Select for kinetic analysis (time vs uptake data)")
        Tooltip(rb2, "Select for isotherm analysis (equilibrium concentration vs uptake)")
        Tooltip(analyze_btn, "Fit the selected models to the loaded data")
        Tooltip(self.result_text, "Fitting results (parameters and statistics)")

    def update_mode(self):
        """Toggle UI elements when switching analysis mode."""
        if self.analysis_mode.get() == "kinetic":
            self.iso_frame.grid_remove()
            self.kin_frame.grid()
        else:
            self.kin_frame.grid_remove()
            self.iso_frame.grid()

    def load_file(self):
        """Load data from a selected file into x_data and y_data."""
        path = filedialog.askopenfilename(
            title="Open Data File",
            filetypes=[("Data files", "*.csv *.txt *.dat"), ("All files", "*.*")]
        )
        if not path:
            return  # user cancelled file dialog
        try:
            # Try using pandas for flexible CSV reading, fall back to numpy
            x_vals, y_vals = None, None
            try:
                import pandas as pd
                # Attempt to read with header inference
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_csv(path, header=None)
                if df.shape[1] < 2:
                    raise ValueError("Data file must have at least two columns")
                x_vals = df.iloc[:, 0].values.astype(float)
                y_vals = df.iloc[:, 1].values.astype(float)
            except ImportError:
                # Fallback to numpy if pandas is not available
                with open(path, 'r') as f:
                    first_line = f.readline().strip()
                skip = 0
                if first_line:
                    try:
                        float(first_line.split(",")[0])
                    except ValueError:
                        skip = 1  # skip header if first value is not numeric
                data = np.loadtxt(path, delimiter=",", skiprows=skip)
                if data.ndim == 1 or data.shape[1] < 2:
                    data = np.loadtxt(path, skiprows=skip)  # try whitespace delimiter
                x_vals = data[:, 0]
                y_vals = data[:, 1]
            # Store loaded data
            self.file_path = path
            self.x_data = x_vals
            self.y_data = y_vals
            # Update UI elements
            self.file_entry.configure(state="normal")
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)
            self.file_entry.configure(state="readonly")
            self.status_var.set(f"Loaded data: {len(x_vals)} points")
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load data: {e}")
            self.status_var.set("Error loading file")

    def analyze_data(self):
        """Fit selected models to the loaded data and update results and plots."""
        # Ensure data is loaded
        if self.x_data is None or self.y_data is None:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return
        # Ensure at least one model is selected
        mode = self.analysis_mode.get()
        if mode == "kinetic":
            selected_models = [m for m, var in self.kin_models_vars.items() if var.get()]
        else:
            selected_models = [m for m, var in self.iso_models_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("No Model Selected", "Please select at least one model to fit.")
            return

        self.status_var.set("Analyzing data...")
        self.root.update_idletasks()  # refresh status display

        x = np.array(self.x_data, dtype=float)
        y = np.array(self.y_data, dtype=float)
        results_output = ""
        model_fits = {}  # to store fit results for plotting and residuals

        # Iterate through selected models and perform fitting
        for model in selected_models:
            try:
                if mode == "kinetic":
                    # --- Kinetic models ---
                    if model == "Pseudo-First Order":
                        # Initial guess: Qe ~ last data point, k1 ~ 0.1
                        Qe0 = y[-1] if len(y) > 0 else 1.0
                        k1_0 = 0.1
                        popt, _ = curve_fit(pfo_func, x, y, p0=[Qe0, k1_0], maxfev=10000) if curve_fit else (None, None)
                        params = popt
                        param_names = ["Qe", "k1"]
                        y_pred = pfo_func(x, *popt) if popt is not None else None
                    elif model == "Pseudo-Second Order":
                        Qe0 = y[-1] if len(y) > 0 else 1.0
                        k2_0 = 0.1
                        popt, _ = curve_fit(pso_func, x, y, p0=[Qe0, k2_0], maxfev=10000) if curve_fit else (None, None)
                        params = popt
                        param_names = ["Qe", "k2"]
                        y_pred = pso_func(x, *popt) if popt is not None else None
                    elif model == "Elovich":
                        # Guess 'a' from initial slope if possible, 'b' as 0.1
                        a0 = 1.0
                        if len(x) > 1:
                            if x[0] == 0 and y[0] == 0 and x[1] > 0:
                                # if first point is (0,0), use second point for slope
                                a0 = (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else 1.0
                            elif x[0] != 0:
                                a0 = y[0] / x[0]
                        b0 = 0.1
                        popt, _ = curve_fit(elovich_func, x, y, p0=[a0, b0], maxfev=10000) if curve_fit else (None, None)
                        params = popt
                        param_names = ["a", "b"]
                        y_pred = elovich_func(x, *popt) if popt is not None else None
                    elif model == "Intraparticle Diffusion":
                        if len(x) < 2:
                            raise RuntimeError("Not enough data points for intraparticle diffusion model")
                        # Fit linear model y = k*sqrt(x) + C
                        sqrt_t = np.sqrt(x)
                        k, C = np.polyfit(sqrt_t, y, 1)
                        params = [k, C]
                        param_names = ["k", "C"]
                        y_pred = k * sqrt_t + C
                    else:
                        continue  # unknown model name (should not happen)
                else:
                    # --- Isotherm models ---
                    if model == "Langmuir":
                        if len(x) < 2:
                            raise RuntimeError("Not enough data for Langmuir fit")
                        # Estimate initial Qmax and KL via linearization: Ce/Qe vs Ce
                        if np.any(y == 0):
                            y_temp = np.where(y == 0, 1e-9, y)  # avoid division by zero
                        else:
                            y_temp = y
                        Ce_over_Qe = x / y_temp
                        m, b = np.polyfit(x, Ce_over_Qe, 1)  # slope and intercept
                        Qmax0 = 1/b if b != 0 else (max(y) if len(y) > 0 else 1.0)
                        KL0 = 1/(Qmax0 * m) if m != 0 else 1.0
                        popt, _ = curve_fit(langmuir_func, x, y, p0=[Qmax0, KL0], maxfev=10000) if curve_fit else ([Qmax0, KL0], None)
                        params = popt
                        param_names = ["Qmax", "KL"]
                        y_pred = langmuir_func(x, *popt)
                    elif model == "Freundlich":
                        if np.any(x <= 0) or np.any(y <= 0):
                            raise RuntimeError("Freundlich model requires positive data")
                        # Estimate initial Kf and n via linearization: log Qe vs log Ce
                        logx = np.log(x)
                        logy = np.log(y)
                        m, b = np.polyfit(logx, logy, 1)  # slope = 1/n, intercept = log(Kf)
                        n0 = 1/m if m != 0 else 1.0
                        Kf0 = math.exp(b)
                        popt, _ = curve_fit(freundlich_func, x, y, p0=[Kf0, n0], maxfev=10000) if curve_fit else ([Kf0, n0], None)
                        params = popt
                        param_names = ["Kf", "n"]
                        y_pred = freundlich_func(x, *popt)
                    else:
                        continue
                # If SciPy's curve_fit was not available (popt is None), raise error to notify user
                if params is None or y_pred is None:
                    raise RuntimeError(f"Could not fit {model} (missing SciPy?)")
                # Compute error metrics
                residuals = y - y_pred
                SSR = np.sum(residuals**2)  # sum of squared residuals
                RMSE = math.sqrt(SSR / len(y))
                k_params = len(params)
                if SSR <= 1e-12:  # avoid log(0)
                    SSR = 1e-12
                AIC = len(y) * math.log(SSR/len(y)) + 2 * k_params
                # Append results to output text
                results_output += f"{model} Model:\n"
                for name, val in zip(param_names, params):
                    results_output += f"  {name} = {val:.4f}\n"
                if self.calc_stats.get():
                    results_output += f"  RMSE = {RMSE:.4f}\n"
                    results_output += f"  AIC = {AIC:.2f}\n"
                results_output += "\n"
                # Save fit results for plotting and residual plot
                model_fits[model] = {"params": params, "residuals": residuals}
            except Exception as e:
                results_output += f"{model} Model: fitting failed ({e})\n\n"

        # Display results in the text area
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, results_output.strip())
        self.result_text.configure(state="disabled")

        # Update plot with data and fits
        self.ax.clear()
        self.ax.scatter(x, y, color='black', label='Data')
        plot_x = np.linspace(x.min(), x.max(), 200) if len(x) > 1 else x
        for model, fit in model_fits.items():
            params = fit["params"]
            if mode == "kinetic":
                if model == "Pseudo-First Order":
                    plot_y = pfo_func(plot_x, *params)
                elif model == "Pseudo-Second Order":
                    plot_y = pso_func(plot_x, *params)
                elif model == "Elovich":
                    plot_y = elovich_func(plot_x, *params)
                elif model == "Intraparticle Diffusion":
                    k, C = params
                    plot_y = k * np.sqrt(plot_x) + C
                else:
                    continue
            else:
                if model == "Langmuir":
                    plot_y = langmuir_func(plot_x, *params)
                elif model == "Freundlich":
                    plot_y = freundlich_func(plot_x, *params)
                else:
                    continue
            self.ax.plot(plot_x, plot_y, label=model)
        if mode == "kinetic":
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Adsorbed Quantity (Qt)")
            self.ax.set_title("Kinetic Model Fit")
        else:
            self.ax.set_xlabel("Equilibrium Concentration (Ce)")
            self.ax.set_ylabel("Adsorbed Quantity (Qe)")
            self.ax.set_title("Isotherm Model Fit")
        self.ax.legend()
        self.canvas.draw()
        self.status_var.set("Analysis complete")

        # Show residual plot in a new window if requested
        if self.show_residual.get() and model_fits:
            res_win = tk.Toplevel(self.root)
            res_win.title("Residual Plot")
            n_models = len(model_fits)
            # determine grid size for subplots
            cols = 1 if n_models <= 2 else 2
            rows = math.ceil(n_models / cols)
            fig2 = plt.Figure(figsize=(5, 4))
            for i, (model, fit) in enumerate(model_fits.items(), start=1):
                ax2 = fig2.add_subplot(rows, cols, i)
                residuals = fit["residuals"]
                ax2.axhline(0, color='gray', linestyle='--')
                ax2.scatter(x, residuals, facecolors='none', edgecolors='blue')
                ax2.set_title(f"{model} Residuals", fontsize=8)
                ax2.set_xlabel("Data X", fontsize=8)
                ax2.set_ylabel("Residual", fontsize=8)
                ax2.tick_params(axis='both', labelsize=8)
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=res_win)
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas2.draw()

    def show_help(self):
        """Show the user guide in a popup."""
        help_text = (
            "Usage:\n"
            "1. Click 'Browse...' to select a CSV or text data file with two columns (time & uptake for kinetics, Ce & Qe for isotherm).\n"
            "2. Choose 'Kinetic' or 'Isotherm' analysis at the top.\n"
            "3. Select one or more models to fit by checking the corresponding boxes.\n"
            "4. (Optional) Enable 'Calculate RMSE & AIC' to compute error metrics and 'Show Residual Plot' to visualize residuals.\n"
            "5. Click 'Analyze Data' to perform the fitting.\n"
            "6. The Results section will display the fitted parameters (and statistics if selected).\n"
            "7. The Plot section will show your data points and the model curves. If residuals are enabled, a new window will show residual plots for each model."
        )
        messagebox.showinfo("User Guide", help_text)

    def show_about(self):
        """Show the about information in a popup."""
        about_text = (
            "Adsorption Data Analyzer v2.0\n"
            "Fits adsorption kinetics (pseudo-first-order, pseudo-second-order, Elovich, intraparticle diffusion)\n"
            "and isotherm models (Langmuir, Freundlich) to experimental data.\n"
            "Provides model parameters and optional fit statistics (RMSE, AIC) and residual diagnostics.\n"
            "Developed for educational and research use, using Python (Tkinter, NumPy, SciPy, Matplotlib)."
        )
        messagebox.showinfo("About", about_text)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AdsorptionApp(root)
    root.mainloop()
