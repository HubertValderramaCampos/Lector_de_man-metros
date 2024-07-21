import cv2
import math
import numpy as np
from scipy.interpolate import interp1d
from ultralytics import YOLO

# Cargar el modelo
model = YOLO("best.pt")

# Definir las clases para la punta de la aguja y el centro del manómetro (ajusta según tu modelo)
clase_aguja = 0  # Reemplaza con el índice de clase correspondiente
clase_centro = 1  # Reemplaza con el índice de clase correspondiente

# Datos de referencia: ángulos y presiones
angulos = np.array([56, 7, 320, 273, 224, 75, 130])
presiones = np.array([0, -5, -10, -15, -20, -25, -30])

# Crear una función de interpolación lineal
interpolador = interp1d(angulos, presiones, kind='linear', fill_value='extrapolate')

# Abrir el video
video_path = 'prueba.mp4'  # Reemplaza con la ruta de tu video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Obtener las propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Cambiar las dimensiones del frame si es necesario (esto es solo un ejemplo, ajusta según sea necesario)
frame_width = 600
frame_height = 600

# Códec para MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Intenta 'XVID' si 'mp4v' no funciona
output_path = 'resultado2.mp4'  # Ruta para guardar el video de salida

# Crear el objeto VideoWriter
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Leer el siguiente frame
    ret, frame = cap.read()

    if not ret:
        break  # Salir si no hay más frames

    # Procesar el frame con el modelo
    results = model(frame)

    # Verificar el tipo de 'results'
    if isinstance(results, list) and results[0] is not None:
        # Acceder a las cajas de detección y extraer información relevante
        boxes = results[0].boxes
        coordenadas = boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
        confianza = boxes.conf.cpu().numpy()  # Confianza de las detecciones
        clases = boxes.cls.cpu().numpy()  # Clases de las detecciones

        # Dibujar las detecciones en el frame
        punto_aguja = None
        punto_centro = None

        for i in range(len(coordenadas)):
            x1, y1, x2, y2 = coordenadas[i]
            conf = confianza[i]
            cls = clases[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if cls == clase_aguja:
                punto_aguja = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            elif cls == clase_centro:
                punto_centro = (int((x1 + x2) // 2), int((y1 + y2) // 2))

        if punto_aguja is not None and punto_centro is not None:
            # Calcular el ángulo en grados en sentido antihorario
            x1, y1 = punto_aguja
            x2, y2 = punto_centro
            angulo = math.degrees(math.atan2(y1 - y2, x1 - x2))

            # Ajustar el ángulo a una referencia de 0 a 360 grados en sentido antihorario desde el eje horizontal
            angulo = (angulo + 360) % 360

            # Dibujar la circunferencia alrededor del centro del manómetro
            radio = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            cv2.circle(frame, punto_centro, radio, (255, 0, 0), 2)

            # Dibujar las marcas de grado y los números correspondientes
            for i in range(0, 360, 10):
                angle_rad = math.radians(i)
                x_out = int(punto_centro[0] + radio * math.cos(angle_rad))
                y_out = int(punto_centro[1] + radio * math.sin(angle_rad))
                x_in = int(punto_centro[0] + (radio - 10) * math.cos(angle_rad))
                y_in = int(punto_centro[1] + (radio - 10) * math.sin(angle_rad))
                cv2.line(frame, (x_in, y_in), (x_out, y_out), (255, 0, 0), 2)

                # Agregar el número de grado en posiciones más legibles
                if i % 30 == 0:
                    text_x = int(punto_centro[0] + (radio + 20) * math.cos(angle_rad))
                    text_y = int(punto_centro[1] + (radio + 20) * math.sin(angle_rad))
                    cv2.putText(frame, f'{i}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # Interpolación para calcular la presión según el ángulo
            presion = interpolador(angulo)

            # Mostrar el ángulo real y el valor de presión en el frame
            cv2.putText(frame, f'Angulo: {angulo:.2f} grados', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f'Presion: {presion:.2f} psi', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # Redimensionar el frame
    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Escribir el frame en el archivo de salida
    out.write(frame_resized)

    # Mostrar el frame con el ángulo y la circunferencia
    cv2.imshow('Deteccion en el video', frame_resized)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la captura y las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()
