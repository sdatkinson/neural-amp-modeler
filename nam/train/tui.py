# File: tui.py
# Created Date: Saturday April 15th 2023
# Author: Tom Persons (blast77@yahoo.com)

"""
TUI for training

Teeps Graphical User Interface

Usage:
>>> from nam.train.tui import run
>>> run()
"""

import os
# Hack to recover graceful shutdowns in Windows.
# This has to happen ASAP
# See:
# https://github.com/sdatkinson/neural-amp-modeler/issues/105
# https://stackoverflow.com/a/44822794
def _ensure_graceful_shutdowns():
    if os.name == "nt":  # OS is Windows
        os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


_ensure_graceful_shutdowns()

import glob
import tkinter as tk
import json
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showerror #, showwarning, showinfo


try:
    from nam import __version__
    from nam.train import core

    _install_is_valid = True
except ImportError:
    _install_is_valid = False

CONFIG_FILE_NAME = "tui_config.json"
FOLDER_ICON = "folder_tiny.png"
BUTTON_WIDTH = 15
BUTTON_HEIGHT = 15
PATH_LABEL_LENGTH = 75
WORD_WRAP_LENGTH = 450
DEBUG = False


SILENT_RUN_KEY  = 'silentrun'
SAVE_PLOT_KEY   = 'saveplot'
TRAINING_EPOCHS_KEY = 'trainingEpochs'
DELAY_KEY = 'delay'
MODEL_NAME_KEY = 'modelName'
OUTPUT_FOLDER_KEY = 'outputFolderName'
INPUT_SOURCE_FILE_KEY = 'inputSourceFile'
SELECTED_ARCH_KEY = 'selectedArchitecture'
CAPTURE_FOLDER_KEY = 'captureFolderName'
SELECTED_AMP_CAPTURE_KEY = 'selectedAmpCapture'
SELECTED_THEME_KEY = 'selectedTheme'


"""
TODO ... 

1. Deal with user entering model name (ie: text entry). 

Currently no hook into return/enter key so settings aren't saved
again until you change another UI setting or click cancel/train.

2. Also need to tie a hook into closing the window via the "X" in the window header in order to save settings
    when closing via this UI mechanism

3. More graceful config file parsing. right now we abort if the config file isn't exactly what we are expecting

4. Multi-file selection and training

"""

class _TUI(ttk.Frame):
    
    def __init__(self, parent):
        ttk.Frame.__init__(self)
        
        config_path = CONFIG_FILE_NAME
        if os.path.exists(config_path):
            with open(config_path) as data_file:
                data_loaded = json.load(data_file)
        else:
            data_loaded = {
                SILENT_RUN_KEY: False,
                SAVE_PLOT_KEY: True,
                TRAINING_EPOCHS_KEY: 100,
                DELAY_KEY: 0,
                MODEL_NAME_KEY: "model.nam",
                OUTPUT_FOLDER_KEY: "",
                INPUT_SOURCE_FILE_KEY: "",
                SELECTED_ARCH_KEY: core.Architecture.FEATHER,
                CAPTURE_FOLDER_KEY: "",
                SELECTED_AMP_CAPTURE_KEY: "",
                SELECTED_THEME_KEY: ""
            }
        
        self.silentrun = tk.BooleanVar(value=data_loaded[SILENT_RUN_KEY])
        self.saveplot = tk.BooleanVar(value=data_loaded[SAVE_PLOT_KEY])
        self.trainingEpochs = tk.IntVar(value=data_loaded[TRAINING_EPOCHS_KEY])
        self.delay = tk.IntVar(value=data_loaded[DELAY_KEY])
        self.modelName = tk.StringVar(value=data_loaded[MODEL_NAME_KEY])
        self.outputFolderName = tk.StringVar(value=data_loaded[OUTPUT_FOLDER_KEY])
        self.inputSourceFile = tk.StringVar(value=data_loaded[INPUT_SOURCE_FILE_KEY])
        self.architectures = [core.Architecture.STANDARD.value,core.Architecture.LITE.value,core.Architecture.FEATHER.value]
        self.selectedArchitecture = tk.StringVar(value=data_loaded[SELECTED_ARCH_KEY])
        self.ampCapturesList = tk.Variable()
        self.captureFolderName = tk.StringVar(value=data_loaded[CAPTURE_FOLDER_KEY])
        self.selectedAmpCapture = ""
        self.selectedTheme = tk.StringVar(value=data_loaded[SELECTED_THEME_KEY])
        self.folder = tk.PhotoImage(file=os.path.join('resources', FOLDER_ICON), width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        
        if data_loaded[CAPTURE_FOLDER_KEY] != "":
            self._parseAmpFolder()

        s = ttk.Style()
        self.systemThemes = s.theme_names()

        self._createLeftAndRightFrames()
        self._createAmpCapturesFrame()
        self._createInputFrame()
        self._createOutputsFrame()
        self._createOptionsFrame()
        self._createThemesFrame()
        self._createTrainingFrame()       

        if self.selectedTheme.get() == "":
            self.selectedTheme.set( self.systemThemes[0] )
            s.theme_use( self.systemThemes[0] )
        else:
            s.theme_use( self.selectedTheme.get() )
        
    
    def _createLeftAndRightFrames(self):
        #left side of window
        self.captureFrame = ttk.LabelFrame( self, text="Amp Captures" )
        self.captureFrame.grid( row=0, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.captureFrame.columnconfigure( index=0, weight=1 )
        self.captureFrame.columnconfigure( index=1, weight=1 )
        self.captureFrame.rowconfigure( index=0, weight=0 )
        self.captureFrame.rowconfigure( index=1, weight=1 )
        
        # Right side of window
        self.rightFrame = ttk.Frame( self )
        self.rightFrame.grid( row=0, column=1, padx=10, pady=10, sticky="nsew" )
        
        self.rightFrame.columnconfigure( index=0, weight=1 )
        self.rightFrame.rowconfigure( index=0, weight=1 )
        self.rightFrame.rowconfigure( index=1, weight=1 )
        self.rightFrame.rowconfigure( index=2, weight=1 )
        self.rightFrame.rowconfigure( index=3, weight=1 )
        self.rightFrame.rowconfigure( index=4, weight=1 )
        
        
    def _createAmpCapturesFrame(self):
        self.captureSelect = ttk.Button( self.captureFrame, image=self.folder, command=self.ampCapturesFolderCallback )
        self.captureSelect.grid( row=0, column=0, padx=10, pady=10, sticky="e", ipadx=5, ipady=5 )
        
        self.captureFolderLabel = ttk.Label( self.captureFrame, textvariable=self.captureFolderName, wraplength=WORD_WRAP_LENGTH, width=PATH_LABEL_LENGTH )
        self.captureFolderLabel.grid( row=0, column=1, padx=10, pady=10, sticky="nsew" )
        
        self.captureList = tk.Listbox( self.captureFrame, listvariable=self.ampCapturesList, exportselection=False )
        self.captureList.grid( row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew" )
        self.captureList.bind('<<ListboxSelect>>', self.ampListSelectionCallback)
        
        
    def _createInputFrame(self):
        # input source 
        self.inputFrame = ttk.LabelFrame( self.rightFrame, text="Input" )
        self.inputFrame.grid( row=0, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.inputFrame.columnconfigure( index=0, weight=1 )
        self.inputFrame.columnconfigure( index=1, weight=0 )
        self.inputFrame.columnconfigure( index=2, weight=2 )
        
        self.inputSourceLabel = ttk.Label( self.inputFrame, text="Input Source:", width=15 )
        self.inputSourceLabel.grid( row=0, column=0, padx=10, pady=10, sticky="e" )
        
        self.inputSourceButton = ttk.Button( self.inputFrame, image=self.folder, command=self.inputSourceFileCallback )
        self.inputSourceButton.grid( row=0, column=1, padx=10, pady=10, sticky="e", ipadx=5, ipady=5 )
        
        self.inputSourceValueLabel = ttk.Label( self.inputFrame, textvariable=self.inputSourceFile, wraplength=WORD_WRAP_LENGTH, width=PATH_LABEL_LENGTH )
        self.inputSourceValueLabel.grid( row=0, column=2, padx=10, pady=10, sticky="nsew" )
        
        
    def _createOutputsFrame(self):
        # outputs 
        self.outputFrame = ttk.LabelFrame( self.rightFrame, text="Outputs" )
        self.outputFrame.grid( row=1, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.outputFrame.columnconfigure( index=0, weight=1 )
        self.outputFrame.columnconfigure( index=1, weight=1 )
        self.outputFrame.columnconfigure( index=2, weight=1 )
        self.outputFrame.rowconfigure( index=0, weight=1 )
        self.outputFrame.rowconfigure( index=1, weight=1 )
        
        self.outputFolderLabel = ttk.Label( self.outputFrame, text="Output Folder:", width=15 )
        self.outputFolderLabel.grid( row=0, column=0, padx=10, pady=10, sticky="e" )
        
        self.outputFolderButton = ttk.Button( self.outputFrame, image=self.folder, command=self.outputFolderCallback )
        self.outputFolderButton.grid( row=0, column=1, padx=10, pady=10, sticky="e", ipadx=5, ipady=5 )
        
        self.outputFolderValueLabel = ttk.Label( self.outputFrame, textvariable=self.outputFolderName, wraplength=WORD_WRAP_LENGTH, width=PATH_LABEL_LENGTH )
        self.outputFolderValueLabel.grid( row=0, column=2, padx=10, pady=10, sticky="nsew" )
        
        self.modelNameLabel = ttk.Label( self.outputFrame, text="Model name", width=15 )
        self.modelNameLabel.grid( row=1, column=0, padx=10, pady=10, sticky="e" )
        
        self.modelNameEntry = ttk.Entry( self.outputFrame, textvariable=self.modelName, width=PATH_LABEL_LENGTH )
        self.modelNameEntry.grid( row=1, column=1, columnspan=2, padx=10, pady=10, sticky="ew" )
        
        
    def _createOptionsFrame(self):
        # options
        self.optionsFrame = ttk.LabelFrame( self.rightFrame, text="Options" )
        self.optionsFrame.grid( row=2, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.optionsFrame.columnconfigure( index=0, weight=1 )
        self.optionsFrame.columnconfigure( index=1, weight=1 )
        self.optionsFrame.rowconfigure( index=0, weight=1 )
        self.optionsFrame.rowconfigure( index=1, weight=1 )
        self.optionsFrame.rowconfigure( index=2, weight=1 )
        self.optionsFrame.rowconfigure( index=3, weight=1 )
        self.optionsFrame.rowconfigure( index=4, weight=1 )
        
        self.epochsLabel = ttk.Label( self.optionsFrame, text="Training Epochs:", width=10 )
        self.epochsLabel.grid( row=0, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.epochsEntry = ttk.Entry( self.optionsFrame, textvariable=self.trainingEpochs, width=3 )
        self.epochsEntry.grid( row=0, column=1, columnspan=2, padx=10, pady=10, sticky="nsew" )
        
        self.delayLabel = ttk.Label( self.optionsFrame, text="Delay:", width=10 )
        self.delayLabel.grid( row=1, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.delayEntry = ttk.Entry( self.optionsFrame, textvariable=self.delay, width=3 )
        self.delayEntry.grid( row=1, column=1, columnspan=2, padx=10, pady=10, sticky="nsew" )
        
        self.archLabel = ttk.Label( self.optionsFrame, text="Architecture:", width=10 )
        self.archLabel.grid( row=2, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.archCombo = ttk.Combobox( self.optionsFrame, textvariable=self.selectedArchitecture, values=self.architectures, width=3, exportselection=False )
        self.archCombo.grid( row=2, column=1, columnspan=2, padx=10, pady=10, sticky="nsew" )
        self.archCombo.bind('<<ComboboxSelected>>', self.archCallback)
        
        self.silentCheckbox = ttk.Checkbutton( self.optionsFrame, text='Silent Run',
                                                command=self.silentrun_changed,
                                                variable=self.silentrun)
        self.silentCheckbox.grid( row=3, column=0, columnspan=3, padx=10, pady=10, sticky="nsew" )
        
        self.savePlotCheckbox = ttk.Checkbutton( self.optionsFrame, text='Save plot automatically',
                                                command=self.saveplot_changed,
                                                variable=self.saveplot)
        self.savePlotCheckbox.grid( row=4, column=0, columnspan=3, padx=10, pady=10, sticky="nsew" )
        
        
    def _createTrainingFrame(self):
        self.executeFrame = ttk.Frame( self.rightFrame )
        self.executeFrame.grid( row=4, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.executeFrame.columnconfigure( index=0, weight=1 )
        self.executeFrame.columnconfigure( index=1, weight=1 )
        self.executeFrame.columnconfigure( index=2, weight=1 )
        self.executeFrame.columnconfigure( index=3, weight=1 )
        self.executeFrame.columnconfigure( index=4, weight=1 )
        self.executeFrame.columnconfigure( index=5, weight=1 )
        
        self.cancelButton = ttk.Button( self.executeFrame, text="Close", command=self.cancelCallback, width=1 )
        self.cancelButton.grid( row=0, column=4, padx=10, pady=10, sticky="nsew" )
        
        self.trainButton = ttk.Button( self.executeFrame, text="Train", command=self.trainCallback, width=1 )
        self.trainButton.grid( row=0, column=5, padx=10, pady=10, sticky="nsew" )


    def _createThemesFrame(self):
         # input source 
        self.themesFrame = ttk.LabelFrame( self.rightFrame, text="Themes" )
        self.themesFrame.grid( row=3, column=0, padx=10, pady=10, sticky="nsew" )
        
        self.themesFrame.columnconfigure( index=0, weight=1 )
        self.themesFrame.columnconfigure( index=1, weight=0 )
        self.themesFrame.columnconfigure( index=2, weight=2 )
        self.themesFrame.rowconfigure( index=0, weight=1 )
        self.themesFrame.rowconfigure( index=1, weight=1 )
        self.themesFrame.rowconfigure( index=2, weight=1 )
        
        self.selectThemeLabel = ttk.Label( self.themesFrame, text="Select Theme:", width=10 )
        self.selectThemeLabel.grid( row=0, column=0, padx=10, pady=10, sticky="nsew" )

        self.themeCombo = ttk.Combobox( self.themesFrame, textvariable=self.selectedTheme, values=self.systemThemes, width=3, exportselection=False )
        self.themeCombo.grid( row=0, column=2, columnspan=2, padx=10, pady=10, sticky="nsew" )
        self.themeCombo.bind('<<ComboboxSelected>>', self.selectThemeCallback)
        

    def silentrun_changed(self):
        if DEBUG: print("silentrun_changed")
        self._saveSettings()
        
    def saveplot_changed(self):
        if DEBUG: print("saveplot_changed")
        self._saveSettings()

    def darkmode_changed(self):
        if DEBUG: print("darkmode_changed")
        if self.darkmode.get():
            self.tk.call("set_theme", "dark")
        else:
            self.tk.call("set_theme", "light")
        self._saveSettings()

    def selectThemeCallback(self, value):
        if DEBUG: print("selectThemeCallback")
        s = ttk.Style()
        s.theme_use( self.selectedTheme.get() )
        
    def archCallback(self, value):
        if DEBUG: print("archCallback: " + self.archCombo.get())
        self._saveSettings()

    def _train(self):
        # TODO... implement multi-file training
        # via multi-file selection in the Amp Capture list
        #file_list = self._path_button_output.val

        # Advanced-er options
        # If you're poking around looking for these, then maybe it's time to learn to
        # use the command-line scripts ;)
        lr = 0.004
        lr_decay = 0.007
        seed = 0

        # Run it
        #for file in file_list:
            #print("Now training {}".format(file))
            #modelname = re.sub(r"\.wav$", "", file.split("/")[-1])

        modelNameNoExt = self.modelName.get().split('.')[0]
       
        trained_model = core.train(
            self.inputSourceFile.get(),
            os.path.join(self.captureFolderName.get(), self.selectedAmpCapture),
            self.outputFolderName.get(),
            epochs=self.trainingEpochs.get(),
            delay=self.delay.get(),
            architecture=self.selectedArchitecture.get(),
            lr=lr,
            lr_decay=lr_decay,
            seed=seed,
            silent=self.silentrun.get(),
            save_plot=self.saveplot.get(),
            modelname=modelNameNoExt,
        )
        print("Model training complete!")
        print("Exporting...")
        outdir = self.outputFolderName.get()
        print(f"Exporting trained model to {outdir}...")
        trained_model.net.export(outdir, modelname=modelNameNoExt)
        print("Done!")
        
    def trainCallback(self):
        if DEBUG: print("trainCallback")
        if DEBUG: print("amp captures folder name: " + self.captureFolderName.get())
        if DEBUG: print("input source file name: " + self.inputSourceFile.get())
        if DEBUG: print("output folder name: " + self.outputFolderName.get())
        if DEBUG: print("model name: " + self.modelName.get())
        if DEBUG: print("num epochs: " + str(self.trainingEpochs.get()))
        if DEBUG: print("delay: " + str(self.delay.get()))
        if DEBUG: print("architecture: " + self.selectedArchitecture.get())
        if DEBUG: print("silent run: " + str(self.silentrun.get()))
        if DEBUG: print("save plot: " + str(self.saveplot.get()))
        if DEBUG: print("selected amp captures: " + self.selectedAmpCapture)
        if DEBUG: print("selected theme: " + str(self.selectedTheme.get()))
        
        self._saveSettings()
        
        # show dialog if things arent set
        if self.selectedAmpCapture == "" or self.inputSourceFile.get() == "" or self.outputFolderName.get() == "":
            showerror(
                title='Error: not ready to train model!',
                message='You must select an amp capture, an input source, and an output folder in order to train models.')
        else:
            self._train()
        
    def cancelCallback(self):
        self._saveSettings()
        # close the window programatically
        self.destroy()
        exit()
        
    def _parseAmpFolder(self):
        listg = glob.glob( "**/*.wav", root_dir=self.captureFolderName.get(), recursive=True )
        if DEBUG: print(listg)
        self.ampCapturesList.set(listg)
        
    def ampCapturesFolderCallback(self):
        if DEBUG: print("ampCapturesFolderCallback")
        # show file picker and then set self.captureFolder to whatever is selected
        self.captureFolderName.set(fd.askdirectory())
        if DEBUG: print("captures folder: " + self.captureFolderName.get())
        self._parseAmpFolder()
        self._saveSettings()
        
    def inputSourceFileCallback(self):
        if DEBUG: print("inputSourceFileCallback")
        # show file picker and then set self.inputSourceFile to whatever is selected
        self.inputSourceFile.set(fd.askopenfilename())
        self._saveSettings()
        
    def outputFolderCallback(self):
        if DEBUG: print("outputFolderCallback")
        # show file picker and then set self.outputFolder to whatever is selected
        self.outputFolderName.set(fd.askdirectory())
        self._saveSettings()
        
    def ampListSelectionCallback(self, event):
        if DEBUG: print("ampListSelectionCallback: "), event
        # get all selected indices
        selected_indices = self.captureList.curselection()
        # get selected items
        selected_items = ",".join([self.captureList.get(i) for i in selected_indices])
        self.selectedAmpCapture = selected_items
        self.modelName.set(os.path.basename(self.selectedAmpCapture).split('.')[0] + ".nam")
        self._saveSettings()
        
    def _saveSettings(self):
        config = {
            SILENT_RUN_KEY: self.silentrun.get(),
            SAVE_PLOT_KEY: self.saveplot.get(),
            TRAINING_EPOCHS_KEY: self.trainingEpochs.get(),
            DELAY_KEY: self.delay.get(),
            MODEL_NAME_KEY: self.modelName.get(),
            OUTPUT_FOLDER_KEY: self.outputFolderName.get(),
            INPUT_SOURCE_FILE_KEY: self.inputSourceFile.get(),
            SELECTED_ARCH_KEY: self.selectedArchitecture.get(),
            CAPTURE_FOLDER_KEY: self.captureFolderName.get(),
            SELECTED_AMP_CAPTURE_KEY: self.selectedAmpCapture,
            SELECTED_THEME_KEY: self.selectedTheme.get()
        }
        with open(CONFIG_FILE_NAME, 'w') as f:
            json.dump(config, f)

def _install_error():
    window = tk.Tk()
    window.title("ERROR")
    label = tk.Label(
        window,
        width=45,
        height=2,
        text="The NAM training software has not been installed correctly.",
    )
    label.pack()
    button = tk.Button(window, width=10, height=2, text="Quit", command=window.destroy)
    button.pack()
    window.mainloop()

def run():
    if _install_is_valid:
        root = tk.Tk()
        root.title("Neural Amp Modeler")
        
        root.columnconfigure(index=0, weight=1)
        root.columnconfigure(index=1, weight=2)
        root.rowconfigure(index=0,weight=1)
        
        app = _TUI(root)
        app.pack(fill="both", expand=True)

        # Set a minsize for the window, and place it in the middle
        root.update()
        root.minsize(root.winfo_width(), root.winfo_height())
        x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
        y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
        root.geometry("+{}+{}".format(x_cordinate, y_cordinate-20))

        root.mainloop()
    else:
        _install_error()


if __name__ == "__main__":
    run()