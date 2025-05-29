import pygame
import pygame_gui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from scipy.optimize import minimize
import tkinter as tk
from tkinter import messagebox

pygame.init()

# Configuración de pantalla más grande
W, H = 1600, 900  # Ventana más grande
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Simulación Avanzada de Llenado de Tanque")
manager = pygame_gui.UIManager((W, H))
clock = pygame.time.Clock()

# Entradas con etiquetas
labels = ["Ancho (m)", "Largo (m)", "Altura (m)", "Caudal Entrada (L/s)", "Caudal Salida (L/s)"]
default_values = ['1.0', '1.0', '2.0', '10.0', '3.0']
inputs = []

for i, (label, val) in enumerate(zip(labels, default_values)):
    y = 50 + i * 70
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, y), (180, 30)), text=label, manager=manager, object_id=f"#label_{i}")
    entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((250, y), (120, 35)), manager=manager)
    entry.set_text(val)
    inputs.append(entry)

# Botones más grandes
btn_start = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((50, 450), (150, 50)), text='Iniciar', manager=manager)
btn_pause = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((220, 450), (150, 50)), text='Pausar', manager=manager)
btn_optimize = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((50, 520), (320, 50)), text='Optimizar Caudales', manager=manager)
btn_results = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((50, 590), (320, 50)), text='Ver Caudales Óptimos', manager=manager)
btn_reset = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((50, 660), (150, 50)), text='Reiniciar', manager=manager)

# Imagen del tanque más grande
tanque_img = pygame.Surface((200, 400), pygame.SRCALPHA)
pygame.draw.rect(tanque_img, (100, 100, 100), (0, 0, 200, 400), 3)
pygame.draw.rect(tanque_img, (150, 150, 150), (5, 5, 190, 390), 1)
tanque_pos = (500, 150)

# Dimensiones internas del área visible del tanque
agua_x = tanque_pos[0] + 20
agua_width = tanque_img.get_width() - 40
agua_height = tanque_img.get_height() - 30
agua_bottom = tanque_pos[1] + tanque_img.get_height() - 15

# Tipografía más grande
font = pygame.font.SysFont(None, 24)
font_large = pygame.font.SysFont(None, 32)

# Variables del sistema
simulating = False
paused = False
fill_level = 0.0
tiempo = 0
dt = 0.1
volumen = 0
alto = 2.0
area_base = 1.0
qin = 10.0
qout = 0.0

# Datos para gráficas
tiempos = []
niveles = []
caudales = []

# Variables para el vaciado cuando está pausado
draining = False
drain_rate = 2.0  # L/s velocidad de vaciado cuando está pausado

# Superficies para gráficas más grandes
graph_surface1 = pygame.Surface((500, 300))
graph_surface2 = pygame.Surface((500, 300))

# Variables para mejores parámetros
best_params = None
optimization_results = []

def update_graphs():
    global graph_surface1, graph_surface2
    
    if len(tiempos) < 2:
        return
    
    # Gráfica 1: Nivel vs Tiempo
    fig, ax = plt.subplots(figsize=(6.25, 3.75))
    ax.plot(tiempos, niveles, 'b-', linewidth=2)
    ax.set_title('Nivel de agua vs Tiempo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tiempo (s)', fontsize=12)
    ax.set_ylabel('Nivel (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_raw(buf)
    buf.seek(0)
    graph_surface1 = pygame.image.frombuffer(buf.getvalue(), fig.canvas.get_width_height(), "RGBA")
    plt.close(fig)
    
    # Gráfica 2: Caudal vs Tiempo
    fig, ax = plt.subplots(figsize=(6.25, 3.75))
    ax.plot(tiempos, caudales, 'r-', linewidth=2)
    ax.set_title('Caudal neto vs Tiempo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tiempo (s)', fontsize=12)
    ax.set_ylabel('Caudal neto (L/s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_raw(buf)
    buf.seek(0)
    graph_surface2 = pygame.image.frombuffer(buf.getvalue(), fig.canvas.get_width_height(), "RGBA")
    plt.close(fig)

    def compute_discharge_time(q_out, area_base, altura):
        "Calcula el tiempo de vaciado considerando salida por gravedad o caudal fijo."
        if q_out <= 0:
            return float('inf')

        volumen_total = area_base * altura  # m³
        tiempo_vaciado = volumen_total / (q_out / 1000)  # q_out está en L/s -> m³/s
        return tiempo_vaciado


def optimize_parameters():
    global best_params, optimization_results

    try:
        optimization_results = []

        def objective(x):
            q_in, q_out = x
            ancho = float(inputs[0].get_text())
            largo = float(inputs[1].get_text())
            h_max = float(inputs[2].get_text())

            area = ancho * largo
            volumen_total = area * h_max

            # Tiempo de llenado (m³ / m³/s)
            caudal_neto = (q_in - q_out) / 1000
            if caudal_neto <= 0:
                return float('inf')
            tiempo_llenado = volumen_total / caudal_neto

            # Tiempo de vaciado (m³ / m³/s)
            if q_out <= 0:
                tiempo_vaciado = float('inf')
            else:
                tiempo_vaciado = volumen_total / (q_out / 1000)

            # Penalización energética (bombeo y control)
            penalty = (q_in / 100) ** 2 + (q_out / 50) ** 2

            # Guardar para análisis
            optimization_results.append({
                'q_in': q_in,
                'q_out': q_out,
                'tiempo_llenado': tiempo_llenado,
                'tiempo_vaciado': tiempo_vaciado,
                'caudal_neto': caudal_neto * 1000,
                'eficiencia_total': volumen_total / (tiempo_llenado + tiempo_vaciado),
                'costo_energetico': penalty
            })

            # Función objetivo: minimizar tiempo total
            return tiempo_llenado + tiempo_vaciado + penalty

        bounds = [(5.0, 100.0), (0.1, 20.0)]
        initial_guess = [20.0, 5.0]

        best_result = None
        best_objective = float('inf')

        for _ in range(10):
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success and result.fun < best_objective:
                best_result = result
                best_objective = result.fun

            initial_guess = [np.random.uniform(10.0, 80.0), np.random.uniform(1.0, 15.0)]

        if best_result and best_result.success:
            ancho = float(inputs[0].get_text())
            largo = float(inputs[1].get_text())
            alto = float(inputs[2].get_text())
            area = ancho * largo
            volumen_total = area * alto
            caudal_neto = best_result.x[0] - best_result.x[1]
            tiempo_llenado = volumen_total / (caudal_neto / 1000)
            tiempo_vaciado = volumen_total / (best_result.x[1] / 1000)

            best_params = {
                'q_in': best_result.x[0],
                'q_out': best_result.x[1],
                'caudal_neto': caudal_neto,
                'tiempo_optimo': best_result.fun,
                'tiempo_llenado': tiempo_llenado,
                'tiempo_vaciado': tiempo_vaciado,
                'ancho': ancho,
                'largo': largo,
                'altura': alto
            }

            inputs[3].set_text(f"{best_result.x[0]:.2f}")
            inputs[4].set_text(f"{best_result.x[1]:.2f}")
            return True

        return False

    except Exception as e:
        print(f"Error en optimización: {e}")
        return False

def show_results_window():
    if not best_params:
        messagebox.showwarning("Advertencia", "Primero debe ejecutar la optimización de caudales.")
        return
    
    # Crear ventana de resultados
    results_window = tk.Tk()
    results_window.title("Caudales Óptimos del Sistema")
    results_window.geometry("600x500")
    results_window.configure(bg='#f0f0f0')
    
    # Título
    title_label = tk.Label(results_window, text="ANÁLISIS DE CAUDALES ÓPTIMOS", 
                          font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
    title_label.pack(pady=20)
    
    # Frame para resultados
    results_frame = tk.Frame(results_window, bg='white', relief='ridge', bd=2)
    results_frame.pack(padx=20, pady=10, fill='both', expand=True)
    
    # Resultados óptimos
    optimal_text = f"""
PARÁMETROS ÓPTIMOS DE CAUDALES ENCONTRADOS:

📐 Dimensiones del Tanque (Fijas):
   • Ancho: {best_params['ancho']:.2f} m
   • Largo: {best_params['largo']:.2f} m
   • Altura: {best_params['altura']:.2f} m
   • Área base: {best_params['ancho'] * best_params['largo']:.2f} m²

💧 Caudales Óptimos:
   • Caudal de ENTRADA: {best_params['q_in']:.2f} L/s
   • Caudal de SALIDA: {best_params['q_out']:.2f} L/s
   • Caudal NETO: {best_params['caudal_neto']:.2f} L/s

⏱️ Eficiencia del Sistema:
   • Tiempo de llenado: {best_params['tiempo_optimo']:.1f} segundos
   • Tiempo en minutos: {best_params['tiempo_optimo']/60:.1f} min
   • Velocidad de llenado: {best_params['altura']/best_params['tiempo_optimo']:.4f} m/s

📊 Análisis de Caudales:
   • Eficiencia hidráulica: {(best_params['caudal_neto']/best_params['q_in'])*100:.1f}%
   • Relación entrada/salida: {best_params['q_in']/max(best_params['q_out'], 0.1):.2f}:1
   • Capacidad neta: {best_params['caudal_neto']:.2f} L/s

💡 Recomendaciones:
   • Estos caudales minimizan el tiempo de llenado
   • Balance óptimo entre velocidad y eficiencia energética
   • Caudal neto optimizado para las dimensiones del tanque
   • Considera costos de bombeo en la optimización
    """
    
    text_widget = tk.Text(results_frame, wrap=tk.WORD, font=('Courier', 11), 
                         bg='white', fg='#2c3e50', padx=20, pady=20)
    text_widget.insert(tk.END, optimal_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Botón cerrar
    close_btn = tk.Button(results_window, text="Cerrar", command=results_window.destroy,
                         font=('Arial', 12), bg='#3498db', fg='white', padx=20)
    close_btn.pack(pady=10)
    
    results_window.mainloop()

def reset_simulation():
    global simulating, paused, fill_level, tiempo, volumen, tiempos, niveles, caudales, draining
    simulating = False
    paused = False
    draining = False
    fill_level = 0.0
    tiempo = 0
    volumen = 0
    tiempos = []
    niveles = []
    caudales = []
    btn_pause.set_text('Pausar')

running = True
while running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == btn_start:
                    try:
                        ancho = float(inputs[0].get_text())
                        largo = float(inputs[1].get_text())
                        alto = float(inputs[2].get_text())
                        qin = float(inputs[3].get_text())
                        qout = float(inputs[4].get_text())
                        area_base = ancho * largo
                        reset_simulation()
                        simulating = True
                        draining = False
                    except ValueError:
                        print("Error: ingrese valores numéricos válidos.")
                        simulating = False

                elif event.ui_element == btn_pause and simulating:
                    paused = not paused
                    draining = paused  # Se vacía solo cuando está pausado
                    btn_pause.set_text('Reanudar' if paused else 'Pausar')
                
                elif event.ui_element == btn_optimize:
                    if optimize_parameters():
                        success_text = "¡Caudales optimizados cargados!"
                        print(success_text)
                
                elif event.ui_element == btn_results:
                    show_results_window()
                
                elif event.ui_element == btn_reset:
                    reset_simulation()

        manager.process_events(event)

    # Lógica de simulación
    if simulating:
        if not paused and fill_level < alto:
            # Llenado normal
            dV = (qin - qout) * dt / 1000
            volumen += dV
            fill_level = volumen / area_base
            fill_level = min(fill_level, alto)
            tiempo += dt
            
            tiempos.append(tiempo)
            niveles.append(fill_level)
            caudales.append(qin - qout)
            
        elif paused and draining and fill_level > 0:
            # Vaciado cuando está pausado
            dV = -drain_rate * dt / 1000
            volumen += dV
            volumen = max(0, volumen)
            fill_level = volumen / area_base
            tiempo += dt
            
            tiempos.append(tiempo)
            niveles.append(fill_level)
            caudales.append(-drain_rate)
        
        # Actualizar gráficas
        if len(tiempos) % 10 == 0:
            update_graphs()

    manager.update(time_delta)

    # =========== DIBUJO ============
    screen.fill((240, 248, 255))  # Color de fondo más agradable

    # Panel de control con mejor diseño
    pygame.draw.rect(screen, (220, 230, 240), (0, 0, 450, H))
    pygame.draw.rect(screen, (180, 190, 200), (0, 0, 450, H), 3)

    # Título del panel
    title_text = font_large.render("CONTROL DE SIMULACIÓN", True, (50, 50, 100))
    screen.blit(title_text, (50, 10))

    if simulating:
        # Dibujar agua en el tanque con mejor visualización
        escala = agua_height / alto
        altura_px = int(fill_level * escala)
        agua_top = agua_bottom - altura_px

        # Gradiente del agua
        for i in range(altura_px):
            alpha = int(200 + (55 * i / max(altura_px, 1)))
            color = (0, min(150 + i//4, 255), 255, min(alpha, 255))
            agua_rect = pygame.Rect(agua_x, agua_top + i, agua_width, 1)
            pygame.draw.rect(screen, color[:3], agua_rect)

        # Dibujar el tanque
        screen.blit(tanque_img, tanque_pos)
        
        # Indicadores de estado
        status_color = (255, 100, 100) if paused else (100, 255, 100)
        status_text = "PAUSADO (VACIANDO)" if paused and draining else "PAUSADO" if paused else "LLENANDO"
        status_surface = font.render(f"Estado: {status_text}", True, status_color)
        screen.blit(status_surface, (500, 580))

        # Información detallada
        info_texts = [
            f'Tiempo: {tiempo:.1f}s',
            f'Nivel: {fill_level:.2f} m / {alto:.1f} m',
            f'Volumen: {volumen:.3f} m³',
            f'Porcentaje: {(fill_level/alto)*100:.1f}%',
            f'Área base: {area_base:.2f} m²'
        ]
        
        for i, text in enumerate(info_texts):
            info_surface = font.render(text, True, (50, 50, 50))
            screen.blit(info_surface, (500, 620 + i * 30))

        # Mostrar gráficas separadas
        if len(tiempos) > 1:
            # Gráfica 1 - Posición superior con más separación
            screen.blit(graph_surface1, (800, 80))
            
            # Gráfica 2 - Posición inferior con más separación
            screen.blit(graph_surface2, (800, 480))
            
            # Títulos de gráficas con mejor posicionamiento
            graph_title1 = font_large.render("📈 Nivel del Tanque", True, (50, 50, 100))
            graph_title2 = font_large.render("📊 Caudal del Sistema", True, (50, 50, 100))
            screen.blit(graph_title1, (800, 45))
            screen.blit(graph_title2, (800, 445))
            
            # Separador visual entre gráficas
            pygame.draw.line(screen, (180, 180, 180), (800, 420), (1300, 420), 2)

    # Mensaje de ayuda actualizado
    help_text = font.render("Tip: Optimiza caudales para mejor eficiencia. El tanque se vacía al pausar.", True, (100, 100, 100))
    screen.blit(help_text, (50, H - 50))

    manager.draw_ui(screen)
    pygame.display.flip()

pygame.quit()