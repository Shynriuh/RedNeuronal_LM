import numpy as np
import sys
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from tkinter.ttk import *

puntos = list()
deseados = list()

def getDotColor(expectedOutput) -> str:
    if expectedOutput == 0:
        return "blue"
    if expectedOutput == 1:
        return "red"

class Window:
    def __init__(self, window):
        style = Style()
        style.configure('W.TButton', font=('calibri', 12, 'bold',
        'underline'),foreground='blue')
        self.window = window
        self.window.title("Proyecto Levenberg-Marquardt")
        self.window.geometry("800x550")
        self.colors = ("red", "blue")
        self.cmap = ListedColormap(self.colors[: len(np.unique([0,1]))])
        self.points = np.zeros((0, 3))
        self.pointsY = np.zeros(0)
        self.entries: Entry = []
        self.loadBtn: Button
        self.classify_btn: Button
        self.restart_btn: Button
        self.MSELabel: Label
        self.figure = None
        self.graph = None
        self.canvas = None
        self.limits = [-9, 9]
        self.x = np.linspace(self.limits[0], self.limits[1], 50)
        self.y = np.linspace(self.limits[0], self.limits[1], 50)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.grafica_conto = np.array([np.ones(len(self.xx.ravel())),
        self.xx.ravel(), self.yy.ravel()]).T
        self.outputs = np.zeros(len(self.grafica_conto))
        self.setup_interface()

    def setup_interface(self) -> None:
        actionFrame = Frame(self.window)
        actionFrame.grid(row=0, column=20, padx=20, ipady=10)
        upperFrame = Frame(actionFrame)
        upperFrame.grid(row=0, column=0)
        Label(upperFrame, text="", width=12).grid(row=5, column=0)
        middleFrame = Frame(actionFrame)
        middleFrame.grid(row=1, column=0)
        self.classify_btn = Button(master=middleFrame, text="Clasificar",
        command=self.start, width=15)
        self.classify_btn.grid(row=1, column=0)
        self.restart_btn = Button(master=middleFrame, text="Reiniciar",
        command=self.restart_all, width=15)
        self.restart_btn.grid(row=2, column=0)
        Label(master=middleFrame, text="", width=12).grid(row=4, column=0)
        lowerFrame = Frame(actionFrame)
        lowerFrame.grid(row=2, column=0)
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.graph = self.figure.add_subplot(111)
        self.config_canvas()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().grid(row=0, column=2)
        cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        
    # Se configura la grafica
    def config_canvas(self) -> None:
        self.graph.cla()
        self.graph.set_xlim([self.limits[0], self.limits[1]])
        self.graph.set_ylim([self.limits[0], self.limits[1]])
        self.graph.axhline(y=0, color="k")
        self.graph.axvline(x=0, color="k")
    
    # Reiniciar interfaz
    def restart_all(self) -> None:
        self.points = np.zeros((0, 3))
        self.pointsY = np.zeros(0)
        puntos.clear()
        deseados.clear()
        self.config_canvas()
        self.canvas.draw()
    
    # Inicio del algoritmo
    def start(self) -> None:
        print("Iniciar clasificación")
        # Implementar algoritmo
        print("Entrenamiento terminado")
    
    # Evento del click para agregar un punto en la grafica
    def on_click(self, event: Event) -> None:
        expected: int
        if event.button == 1:
            puntos.append([event.xdata, event.ydata])
            deseados.append(0)
            expected = 0
        elif event.button == 3:
            puntos.append([event.xdata, event.ydata])
            deseados.append(1)
            expected = 1
        
        # Los agregamos la array de puntos
        self.points = np.append(
            self.points, [[1, float(event.xdata), float(event.ydata)]], axis=0)
        # Agregamos la salida yD correspondiente
        self.pointsY = np.append(self.pointsY, [expected])
        # Dibujamos el punto
        self.graph.plot(
            event.xdata,
            event.ydata,
            marker="o",
            c=getDotColor(expected),
        )
        self.canvas.draw()

def main() -> int:
    window = Tk()
    window.config(bg="gray", bd="10", relief="groove", pady=10, padx=10)
    app = Window(window)
    window.mainloop()
    return 0

if __name__ == '__main__':
    sys.exit(main())