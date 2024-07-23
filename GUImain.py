import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import importlib.util
import os
import glob

class MainApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Optimization Problem Manager")

        # Tab Control
        self.tab_control = ttk.Notebook(master)
        self.create_dataset_tab = ttk.Frame(self.tab_control)
        self.add_script_tab = ttk.Frame(self.tab_control)
        self.evaluate_algorithms_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.create_dataset_tab, text="Create Dataset")
        self.tab_control.add(self.add_script_tab, text="Add Python Script")
        self.tab_control.add(self.evaluate_algorithms_tab, text="Evaluate Algorithms")

        self.tab_control.pack(expand=1, fill="both")

        # Create Dataset Tab
        self.dataset_label = tk.Label(self.create_dataset_tab, text="Enter Data:")
        self.dataset_label.pack()
        self.dataset_entry = tk.Text(self.create_dataset_tab, height=10)
        self.dataset_entry.pack()
        self.save_dataset_button = tk.Button(self.create_dataset_tab, text="Save Dataset", command=self.save_dataset)
        self.save_dataset_button.pack()

        # Add Python Script Tab
        self.script_label = tk.Label(self.add_script_tab, text="Upload Python Script:")
        self.script_label.pack()
        self.upload_script_button = tk.Button(self.add_script_tab, text="Upload Script", command=self.upload_script)
        self.upload_script_button.pack()
        self.run_script_button = tk.Button(self.add_script_tab, text="Run Script", command=self.run_script)
        self.run_script_button.pack()
        self.script_output = tk.Text(self.add_script_tab, height=10)
        self.script_output.pack()

        # Evaluate Algorithms Tab
        self.algorithm_label = tk.Label(self.evaluate_algorithms_tab, text="Select Algorithm:")
        self.algorithm_label.pack()
        self.algorithm_combo = ttk.Combobox(self.evaluate_algorithms_tab)
        self.algorithm_combo.pack()
        self.load_algorithms()
        self.evaluate_button = tk.Button(self.evaluate_algorithms_tab, text="Evaluate", command=self.evaluate_algorithm)
        self.evaluate_button.pack()
        self.evaluation_output = tk.Text(self.evaluate_algorithms_tab, height=10)
        self.evaluation_output.pack()

    def save_dataset(self):
        data = self.dataset_entry.get("1.0", tk.END)
        if data.strip():
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(data)
                messagebox.showinfo("Success", f"Dataset saved to {file_path}")
        else:
            messagebox.showerror("Error", "No data entered.")

    def upload_script(self):
        file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if file_path:
            script_name = os.path.basename(file_path)
            target_path = os.path.join('scripts', script_name)
            os.makedirs('scripts', exist_ok=True)
            with open(file_path, 'r') as src_file:
                with open(target_path, 'w') as dst_file:
                    dst_file.write(src_file.read())
            messagebox.showinfo("Success", f"Script uploaded to {target_path}")

    def run_script(self):
        script_files = glob.glob('scripts/*.py')
        if script_files:
            script_name = os.path.basename(script_files[0])
            spec = importlib.util.spec_from_file_location(script_name, script_files[0])
            script_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(script_module)
            self.script_output.insert(tk.END, f"Script {script_name} executed successfully.\n")
        else:
            messagebox.showerror("Error", "No scripts found to run.")

    def load_algorithms(self):
        # Placeholder for loading algorithms - could be extended to load dynamically
        self.algorithm_combo['values'] = ('Algorithm1', 'Algorithm2', 'Algorithm3')

    def evaluate_algorithm(self):
        selected_algorithm = self.algorithm_combo.get()
        if selected_algorithm:
            self.evaluation_output.insert(tk.END, f"Evaluating {selected_algorithm}...\n")
            # Placeholder for actual evaluation logic
            self.evaluation_output.insert(tk.END, f"Results for {selected_algorithm}: ...\n")
        else:
            messagebox.showerror("Error", "No algorithm selected.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
