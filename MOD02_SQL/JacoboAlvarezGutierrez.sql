-- Jacobo Álvarez Gutiérrez
-- Proyecto Final: "ArteVida Cultural"

-- Borrar database si ya existía
DROP DATABASE IF EXISTS ArteVidaCultural;

-- Creación de la base de datos
CREATE DATABASE ArteVidaCultural;
USE ArteVidaCultural;

-- Creación de las tablas correspondientes al modelo conceptual
CREATE TABLE actividades(
IdActividad CHAR(5),
NombreAc VARCHAR(255) NOT NULL,
TipoAc VARCHAR(255) NOT NULL,
PRIMARY KEY (IdActividad)
);

CREATE TABLE artistas(
IdArtista CHAR(5),
NombreArt VARCHAR(255) NOT NULL,
Biografia TEXT,
PRIMARY KEY (IdArtista)
);

CREATE TABLE ubicaciones(
IdUbicacion CHAR(5),
NombreUb VARCHAR(255) NOT NULL,
Direccion VARCHAR(255) NOT NULL,
Tipologia ENUM('Pueblo', 'Ciudad') NOT NULL,
Aforo INT NOT NULL,
Alquiler DECIMAL(10,2) NOT NULL,
Caracteristicas TEXT,
PRIMARY KEY (IdUbicacion)
);

CREATE TABLE personas(
DNI CHAR(9),
NombrePer VARCHAR(255) NOT NULL,
Telefono VARCHAR(20),
email VARCHAR(255) NOT NULL,
PRIMARY KEY (DNI)
);

CREATE TABLE eventos(
IdEvento CHAR(5),
NombreEvento VARCHAR(225) NOT NULL,
FechaHora DATETIME NOT NULL,
Descripcion TEXT,
IdActividad CHAR(5),
IdUbicacion CHAR(5),
PRIMARY KEY (IdEvento),
FOREIGN KEY (IdActividad) REFERENCES actividades(IdActividad),
FOREIGN KEY (IdUbicacion) REFERENCES ubicaciones(IdUbicacion)
);

/*
ON DELETE CASCADE aquí hace que, si se elimina un cierto evento,
también se eliminen los registros de asistencia al mismo.
*/
CREATE TABLE asisten(
DNI CHAR(9),
IdEvento CHAR(5),
PRIMARY KEY (DNI, IdEvento),
FOREIGN KEY (DNI) REFERENCES personas(DNI),
FOREIGN KEY (IdEvento) REFERENCES eventos(IdEvento) ON DELETE CASCADE
);

CREATE TABLE participan(
IdArtista CHAR(5),
IdActividad CHAR(5),
IdEvento CHAR(5),
Honorarios DECIMAL(10,2) NOT NULL,
PRIMARY KEY (IdArtista, IdActividad, IdEvento),
FOREIGN KEY (IdArtista) REFERENCES artistas(IdArtista),
FOREIGN KEY (IdActividad) REFERENCES actividades(IdActividad),
FOREIGN KEY (IdEvento) REFERENCES eventos(IdEvento) ON DELETE CASCADE
);

/*
Incluimos un trigger para que no puedan incluirse asistencias de personas
a eventos que aún no han tenido lugar.
*/
DELIMITER //

CREATE TRIGGER prevent_future_assistance BEFORE INSERT ON asisten FOR EACH ROW
BEGIN
    IF (SELECT FechaHora FROM eventos WHERE IdEvento = NEW.IdEvento) > NOW()
    THEN SIGNAL SQLSTATE '45000'
	SET MESSAGE_TEXT = 'No se pueden registrar asistencias para 
    eventos futuros';
	END IF;
END;
//

/*
Añadimos otro trigger para asegurarnos de que no se supere el aforo
del local en cada evento.
*/
CREATE TRIGGER prevent_overcapacity BEFORE INSERT ON asisten FOR EACH ROW
BEGIN
    IF (SELECT COUNT(DISTINCT DNI) FROM asisten
        WHERE IdEvento = NEW.IdEvento) >=
	   (SELECT u.Aforo FROM eventos AS e INNER JOIN ubicaciones AS u
        ON e.IdUbicacion = u.IdUbicacion WHERE e.IdEvento = NEW.IdEvento)
	   THEN SIGNAL SQLSTATE '45000'
       SET MESSAGE_TEXT = 'No se pueden registrar más asistentes: 
       Aforo completo';
	END IF;
END;
//

DELIMITER ;

-- Insertamos algunos datos de prueba
INSERT INTO actividades VALUES 
('A0001', 'Concierto de Jazz', 'Música'),
('A0002', 'Exposición de Pintura', 'Arte Visual'),
('A0003', 'Obra de Teatro', 'Teatro'),
('A0004', 'Conferencia sobre Literatura', 'Conferencia'),
('A0005', 'Festival de Cine', 'Cine');

INSERT INTO artistas VALUES
('AR001', 'Carlos López', 'Saxofonista con 20 años de experiencia en jazz 
y blues.'),
('AR002', 'Ana Martín', 'Pintora contemporánea reconocida 
internacionalmente.'),
('AR003', 'Jorge Ruiz', 'Actor de teatro y televisión.'),
('AR004', 'María Torres', 'Escritora y conferencista sobre 
literatura española.'),
('AR005', 'Laura Sánchez', 'Directora de cine especializada en 
documentales.'),
('AR006', 'Miguel García', 'Pianista de jazz colaborador de varios 
artistas.'),
('AR007', 'Elena Vidal', 'Fotógrafa y artista visual innovadora.');

INSERT INTO ubicaciones VALUES
('U0001', 'Teatro Principal', 'Calle Mayor 123, Madrid', 'Ciudad', 300,
1200.00, 'Teatro con acústica excelente y asientos cómodos.'),
('U0002', 'Centro de Arte Moderno', 'Av. del Arte 45, Barcelona', 'Ciudad',
200, 900.00, 'Sala de exposiciones con iluminación profesional.'),
('U0003', 'Sala Cultural', 'Plaza del Pueblo, Valencia', 'Pueblo', 100,
500.00, 'Espacio para eventos pequeños y medianos.'),
('U0004', 'Auditorio Nacional', 'Paseo de la Reforma 50', 'Ciudad', 1000,
5000.00, 'Gran auditorio con sonido envolvente'),
('U0005', 'Cine Capitol', 'Gran Vía 42, Madrid', 'Ciudad', 500, 1500.00,
'Sala de cine histórica.');

INSERT INTO personas VALUES
('12345678A', 'Luis Gómez', '600123456', 'luis.gomez@example.com'),
('23456789B', 'Marta Pérez', '650234567', 'marta.perez@example.com'),
('34567890C', 'Juan Ruiz', '620345678', 'juan.ruiz@example.com'),
('45678901D', 'Ana Torres', '630456789', 'ana.torres@example.com'),
('56789012E', 'Carlos Sánchez', '610567890', 'carlos.sanchez@example.com'),
('67890123F', 'Lucía Fernández', '615678901', 'lucia.fernandez@example.com'),
('78901234G', 'Javier López', '620789012', 'javier.lopez@example.com');

INSERT INTO eventos VALUES
('E0001', 'Noche de Jazz', '2023-11-20 20:00:00',
'Concierto de jazz con artistas reconocidos', 'A0001', 'U0001'),
('E0002', 'Exposición Vanguardista', '2023-12-01 10:00:00',
'Muestra de arte contemporáneo', 'A0002', 'U0002'),
('E0003', 'Teatro Clásico', '2023-12-15 19:00:00',
'Representación de una obra clásica', 'A0003', 'U0003'),
('E0004', 'Conferencia Literaria', '2023-11-25 18:00:00',
'Conferencia sobre la literatura española', 'A0004', 'U0004'),
('E0005', 'Festival de Documentales', '2023-01-10 17:00:00',
'Festival con los mejores documentales del año', 'A0005', 'U0005'),
('E0006', 'Jazz y Poesía', '2023-11-30 20:30:00',
'Fusión de música jazz y recital de poesía', 'A0001', 'U0001'),
('E0007', 'Festival de Arte Digital', '2023-02-15 11:00:00',
'Exposición de arte digital contemporáneo', 'A0002', 'U0002'),
('E0008', 'Gala de Jazz', '2025-03-20 20:00:00',
'Concierto de gala con destacados músicos de jazz', 'A0001', 'U0001'),
('E0009', 'Retrospectiva de Pintura', '2025-04-15 18:00:00',
'Exposición retrospectiva de arte contemporáneo', 'A0002', 'U0002'),
('E0010', 'Obra de Teatro Moderna', '2025-05-05 19:30:00',
'Representación teatral moderna', 'A0003', 'U0003'),
('E0011', 'Congreso Literario', '2025-06-01 10:00:00',
'Congreso internacional sobre literatura', 'A0004', 'U0004'),
('E0012', 'Festival Internacional de Cine', '2025-07-10 16:00:00',
'Festival con proyecciones de cine internacional', 'A0005', 'U0005');

INSERT INTO participan VALUES
('AR001', 'A0001', 'E0001', 800.00),
('AR001', 'A0001', 'E0006', 850.00),
('AR001', 'A0001', 'E0008', 1000.00),
('AR002', 'A0002', 'E0002', 600.00),
('AR002', 'A0002', 'E0007', 650.00),
('AR002', 'A0002', 'E0009', 700.00),
('AR003', 'A0003', 'E0003', 700.00),
('AR003', 'A0003', 'E0010', 750.00),
('AR004', 'A0004', 'E0004', 500.00),
('AR004', 'A0004', 'E0011', 550.00),
('AR005', 'A0005', 'E0005', 1000.00),
('AR005', 'A0005', 'E0012', 1100.00),
('AR006', 'A0001', 'E0001', 750.00),
('AR006', 'A0001', 'E0008', 950.00);

INSERT INTO asisten VALUES
('12345678A', 'E0001'),
('23456789B', 'E0001'),
('34567890C', 'E0002'),
('45678901D', 'E0003'),
('56789012E', 'E0004'),
('67890123F', 'E0005'),
('78901234G', 'E0006'),
('12345678A', 'E0006'),
('23456789B', 'E0007'),
('67890123F', 'E0007');

/* 
A continuación, haremos diversas consultas. Estas irán aumentando su complejidad
sucesivamente.
*/

-- NÚMERO DE COLABORACIONES DE CADA ARTISTA
SELECT a.IdArtista, NombreArt, COUNT(IdEvento) AS num_collabs
FROM participan AS p INNER JOIN artistas AS a ON  p.IdArtista=a.IdArtista
GROUP BY a.IdArtista
ORDER BY num_collabs DESC;

-- LISTADO DEL NÚMERO DE EVENTOS POR CIUDAD
SELECT e.IdUbicacion, Direccion, COUNT(IdEvento) AS num_eventos
FROM ubicaciones AS u INNER JOIN eventos AS e
ON u.IdUbicacion=e.IdUbicacion
GROUP BY e.IdUbicacion, Direccion
ORDER BY num_eventos DESC;

-- ARTISTAS QUE HAN PARTICIPADO EN MÁS DE UN EVENTO
SELECT a.IdArtista, a.NombreArt, COUNT(p.IdEvento) AS NumEventos
FROM artistas AS a
INNER JOIN participan AS p ON a.IdArtista = p.IdArtista
GROUP BY a.IdArtista, a.NombreArt
HAVING COUNT(p.IdEvento) > 1
ORDER BY NumEventos DESC;

-- CANTIDAD DE ASISTENTES A CADA EVENTO
SELECT asi.IdEvento, NombreEvento, COUNT(DISTINCT(DNI)) AS num_personas FROM
eventos AS e INNER JOIN asisten AS asi ON
e.IdEvento = asi.IdEvento
GROUP BY asi.IdEvento;

-- EVENTOS QUE YA TUVIERON LUGAR Y SU PORCENTAJE DE OCUPACIÓN EN RELACIÓN CON
-- EL AFORO DE LA UBICACIÓN
/*
En este caso debemos usar un LEFT JOIN al unir las tablas evento y asisten.
Esto se hace así para poder contemplar la posibilidad de que no haya ido
ninguna persona a alguno de los eventos realizados.
*/
SELECT e.IdEvento, NombreEvento, count(distinct(asi.DNI)) AS num_personas,
Aforo, CONCAT(ROUND((COUNT(DISTINCT asi.DNI)/Aforo)*100, 2), '%')
AS porcentaje_ocupacion FROM eventos AS e 
LEFT JOIN asisten AS asi ON e.IdEvento=asi.IdEvento
INNER JOIN ubicaciones AS u ON e.IdUbicacion=u.IdUbicacion
WHERE e.FechaHora <= NOW()
GROUP BY e.IdEvento, NombreEvento, Aforo;
/*
Evidentemente, los resultados de esta consulta no son del todo significativos
ya que hablamos de aforos de en torno al millar de personas y apenas hemos
puesto 1 o 2 personas por evento. No obstante, nos sirve para comprobar que
la consulta está bien planteada.
*/

-- LISTADO DE ARTISTAS Y SU PROMEDIO DE HONORARIOS POR LOS EVENTOS YA
-- REALIZADOS
SELECT a.NombreArt, ROUND(AVG(p.Honorarios), 2) AS PromedioHonorarios
FROM artistas AS a
INNER JOIN participan AS p ON a.IdArtista = p.IdArtista
INNER JOIN eventos AS e ON p.IdEvento = e.IdEvento
WHERE e.FechaHora <= NOW()
GROUP BY a.IdArtista, a.NombreArt
ORDER BY PromedioHonorarios DESC;

-- PRECIO DE LAS ENTRADAS (VISTA)
CREATE VIEW PrecioEntradas AS
SELECT e.IdEvento, e.NombreEvento, u.Aforo, 
ROUND((u.Alquiler + COALESCE(SUM(p.honorarios), 0))*1.2/u.Aforo, 2)
AS PrecioEntrada FROM eventos AS e
INNER JOIN ubicaciones AS u ON e.IdUbicacion = u.IdUbicacion
LEFT JOIN participan AS p ON e.IdEvento = p.IdEvento
GROUP BY e.IdEvento, e.NombreEvento, u.Aforo;

-- CONSULTA DE LA VISTA ANTERIOR PARA OBTENER LOS RESULTADOS POR PANTALLA
SELECT IdEvento, NombreEvento, PrecioEntrada FROM PrecioEntradas;

-- EVENTOS DONDE EL PRECIO DE LA ENTRADA ES MENOR A UN VALOR ESPECÍFICO
SELECT pe.IdEvento, pe.NombreEvento, CONCAT(pe.Precioentrada, '€') AS Entrada
FROM PrecioEntradas AS pe
WHERE pe.PrecioEntrada < 10
ORDER BY pe.PrecioEntrada ASC;

-- LISTADO DE EVENTOS REALIZADOS Y EL BENEFICIO OBTENIDO POR LA EMPRESA
SELECT e.IdEvento, e.NombreEvento,
count(distinct asi.DNI)*PrecioEntrada
-(u.Alquiler + COALESCE(SUM(p.honorarios), 0))
AS beneficio FROM eventos AS e
INNER JOIN asisten AS asi ON e.IdEvento=asi.IdEvento
INNER JOIN PrecioEntradas AS pe ON e.IdEvento=pe.IdEvento
INNER JOIN ubicaciones AS u ON e.IdUbicacion = u.IdUbicacion
LEFT JOIN participan AS p ON e.IdEvento = p.IdEvento
WHERE e.FechaHora <= NOW()
GROUP BY e.IdEvento, e.NombreEvento, u.Alquiler
ORDER BY beneficio DESC;
/*
Aquí los resultados vuelven a no ser del todo significativos por el mismo
problema con el aforo. El beneficio de la empresa será negativo puesto que,
a modo de prueba, solo se han añadido 1 o 2 asistentes por eventos, mientras
que los valores reales que deberían barajarse aquí es en torno al millar de
personas. No obstante, nos sirve para comprobar que la consulta se lleva a 
cabo de forma correcta.
*/

-- LISTADO CON EL PROMEDIO DEL PORCENTAJE DE ASISTENCIA POR CADA TIPO DE
-- ACTIVIDAD
/*
Al igual que ocurría en otra consulta previa, aquí necesitamos hacer un 
LEFT JOIN entre eventos y la subconsulta para considerar la posibilidad
de registros de actividades sin ningún asistente al evento.
*/
SELECT a.TipoAc, ROUND(AVG(evento_porcentajes.porcentaje_asistencia), 2)
AS promedio_porcentaje_asistencia FROM actividades AS a 
INNER JOIN eventos AS e ON a.IdActividad = e.IdActividad
LEFT JOIN (
    SELECT e.IdEvento, (COUNT(asi.DNI)/u.Aforo)*100 AS porcentaje_asistencia
    FROM eventos AS e INNER JOIN ubicaciones AS u
    ON e.IdUbicacion = u.IdUbicacion
    LEFT JOIN asisten AS asi ON e.IdEvento = asi.IdEvento
    WHERE e.FechaHora <= NOW()
    GROUP BY e.IdEvento, u.aforo)
AS evento_porcentajes ON e.IdEvento = evento_porcentajes.IdEvento
GROUP BY a.TipoAc
ORDER BY promedio_porcentaje_asistencia DESC;
/*
Ídem que en las dos consultas anteriores que hacen referencia al porcentaje
de ocupación.
*/