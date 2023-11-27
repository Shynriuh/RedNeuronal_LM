import numpy as np
import sys
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from tkinter.ttk import *

puntos = list()
deseados = list()

x = np.arange(-9, 9, 0.1)
y = np.arange(-9, 9, 0.1)
xx, yy = np.meshgrid(x, y)
grafica_conto = np.c_[xx.ravel(), yy.ravel()]

def getDotColor(expectedOutput) -> str:
    if expectedOutput == 0:
        return "blue"
    if expectedOutput == 1:
        return "red"
# Funcion de activacion
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Matriz Jacobiana
def compute_jacobian(X, w1, w2, b1, b2, a1, a2):
    num_samples = X.shape[0]
    num_w1 = np.prod(w1.shape)
    num_w2 = np.prod(w2.shape)
    num_b1 = b1.size
    num_b2 = b2.size

    J = np.zeros((num_samples, num_w1 + num_w2 + num_b1 + num_b2))

    for i in range(num_samples):
        # Se deriva la funcion sigmoideal
        sig_prime_a1 = a1[i] * (1 - a1[i])
        sig_prime_a2 = a2[i] * (1 - a2[i])

        # Se deriva la capa de salida
        d_a2_d_w2 = a1[i, :, np.newaxis] * sig_prime_a2[:, np.newaxis]
        d_a2_d_b2 = sig_prime_a2

        # Se deriva la capa oculta
        d_a1_d_w1 = X[i, :, np.newaxis] * sig_prime_a1[:, np.newaxis].T
        d_a1_d_b1 = sig_prime_a1

        # Se construye la matriz Jacobiana
        J[i, :num_w1] = d_a1_d_w1.flatten()
        J[i, num_w1:num_w1 + num_b1] = d_a1_d_b1
        J[i, num_w1 + num_b1:num_w1 + num_b1 + num_w2] = d_a2_d_w2.flatten()
        J[i, -num_b2:] = d_a2_d_b2

    return J

def extract_weight_updates(weight_update, w1, w2, b1, b2):
    num_w1 = np.prod(w1.shape)
    num_w2 = np.prod(w2.shape)
    num_b1 = b1.size
    num_b2 = b2.size

    update_w1 = weight_update[:num_w1].reshape(w1.shape)
    update_b1 = weight_update[num_w1:num_w1 + num_b1].reshape(b1.shape)
    update_w2 = weight_update[num_w1 + num_b1:num_w1 + num_b1 + num_w2].reshape(w2.shape)
    update_b2 = weight_update[-num_b2:].reshape(b2.shape)

    return update_w1, update_w2, update_b1, update_b2

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
        if(len(puntos) > 0):
            puntosnp = []
            deseadosnp = []
            for i in range(len(puntos)):
                puntosnp.append([puntos[i][0], puntos[i][1]])
                deseadosnp.append(deseados[i])
            nppuntos = np.array(puntosnp)
            npdeseados = np.array(deseadosnp)
        X = nppuntos
        y = npdeseados.reshape(len(npdeseados),1)
        self.w1 = np.random.random((2, 3))
        self.b1 = np.zeros((1, 3))
        self.w2 = np.random.random((3, 1))
        self.b2 = np.zeros((1, 1))
        self.learning_rate = 0.1
        iter = 2000
        for i in range(iter):
            salidaO = sigmoid(np.dot(X, self.w1) + self.b1)
            salidaS = sigmoid(np.dot(salidaO, self.w2) + self.b2)
            salidaOGr = sigmoid(np.dot(grafica_conto, self.w1) + self.b1)
            salidaSGr = sigmoid(np.dot(salidaOGr, self.w2) + self.b2)
            z = salidaSGr.reshape(xx.shape)
            self.window.update()
            self.config_canvas()
            for i in range(len(self.points)):
                self.graph.plot(
                    self.points[i][1],
                    self.points[i][2],
                    marker="o",
                    c=getDotColor(self.pointsY[i]),
                )
            self.graph.contourf(xx, yy, z, cmap="seismic")
            self.canvas.draw()
            error = y - salidaS
            delta = error * (salidaS * (1 - salidaS))
            deltaO = (salidaO * (1 - salidaO)) * delta * self.w2.T
            self.w2 += salidaS.T.dot(delta) * self.learning_rate
            self.b2 += np.sum(delta, axis=0, keepdims=True) * self.learning_rate
            self.w1 += X.T.dot(deltaO) * self.learning_rate
            self.b1 += np.sum(deltaO, axis=0, keepdims=True) * self.learning_rate
        print("Terminado")
    
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
