// EJERCICIO 0
use "MOVIES"

// EJERCICIO 1
db.movies.find()

// EJERCICIO 2
db.movies.find().count()

// EJERCICIO 3
var nuevo_item = {"title": "Película de prueba", "year": "2050", "cast": [], "genres": []}
db.movies.insertOne(nuevo_item)

// EJERCICIO 4
var nuevo_item = {"title": "Película de prueba", "year": "2050", "cast": [], "genres": []}
db.movies.deleteOne(nuevo_item)

// EJERCICIO 5
var query = {"cast": "and"}
db.movies.find(query).count()

// EJERCICIO 6
var query = {"cast": "and"}
var operation = {$pull: query}
db.movies.updateMany(query, operation)

// EJERCICIO 7
var query1 = {"cast": {$size: 0}}
var query2 = {"cast": {$exists: false}}
var queryT = {$or: [query1, query2]}
db.movies.find(queryT).count()

// EJERCICIO 8
var query1 = {"cast": {$size: 0}}
var query2 = {"cast": {$exists: false}}
var queryT = {$or: [query1, query2]}
var operation = {$set: {"cast": ["Undefined"]}}
db.movies.updateMany(queryT, operation)

// EJERCICIO 9
var query1 = {"genres": {$size: 0}}
var query2 = {"genres": {$exists: false}}
var queryT = {$or: [query1, query2]}
db.movies.find(queryT).count()

// EJERCICIO 10
var query1 = {"genres": {$size: 0}}
var query2 = {"genres": {$exists: false}}
var queryT = {$or: [query1, query2]}
var operation = {$set: {"genres": ["Undefined"]}}
db.movies.updateMany(queryT, operation)
db.movies.find()

// EJERCICIO 11
var query = {}
var projection = {"_id": 0, "year": 1}
db.movies.find(query, projection).sort({"year": -1}).limit(1)

// EJERCICIO 12
var query1 = {"_id": "$year", "num_peliculas": {$sum: 1}}
var fase1 = {$group: query1}
var query2 = {"_id": -1}
var fase2 = {$sort: query2}
var fase3 = {$limit: 20}
var query4 = {"_id": null, "total": {$sum: "$num_peliculas"}}
var fase4 = {$group: query4}
var etapas = [fase1, fase2, fase3, fase4]
db.movies.aggregate(etapas)
// El ejercicio anterior también se podría haber hecho como el 13
// Es decir, buscando las películas de los últimos 20 años en total
// Y sumándolas directamente, sin hacer una fase previa para ver cuantas
// hay cada año.

// EJERCICIO 13
var query1 = {"year": {$gte: 1960, $lte: 1969}}
var fase1 = {$match: query1}
var query2 = {"_id": null, "total": {$sum: 1}}
var fase2 = {$group: query2}
var etapas = [fase1, fase2]
db.movies.aggregate(etapas)

// EJERCICIO 14
var query1 = {"_id": "$year", "num_peliculas": {$sum: 1}}
var fase1 = {$group: query1}
var fase2 = {$sort: {"num_peliculas": -1}}
var fase3 = {$limit: 1}
var etapas1 = [fase1, fase2, fase3]
var maxpelis = db.movies.aggregate(etapas1).toArray()[0].num_peliculas
var query4 = {"num_peliculas": {$eq: maxpelis}}
var fase4 = {$match: query4}
var etapas2 = [fase1, fase4]
db.movies.aggregate(etapas2)

// EJERCICIO 15
var query1 = {"_id": "$year", "num_peliculas": {$sum: 1}}
var fase1 = {$group: query1}
var fase2 = {$sort: {"num_peliculas": 1}}
var fase3 = {$limit: 1}
var etapas1 = [fase1, fase2, fase3]
var minpelis = db.movies.aggregate(etapas1).toArray()[0].num_peliculas
var query4 = {"num_peliculas": {$eq: minpelis}}
var fase4 = {$match: query4}
var etapas2 = [fase1, fase4]
db.movies.aggregate(etapas2)

// EJERCICIO 16
var fase1 = {$unwind: "$cast"}
var query2 = {"_id": 0}
var fase2 = {$project: query2}
var fase3 = {$out: "actors"}
var etapas = [fase1, fase2, fase3]
db.movies.aggregate(etapas)
db.actors.count()

// EJERCICIO 17
var query1 = {"_id": "$cast", "num_peliculas": {$sum: 1}}
var fase1 = {$group: query1}
var query2 = {"_id": {$ne: "Undefined"}}
var fase2 = {$match: query2}
var fase3 = {$sort: {"num_peliculas": -1}}
var fase4 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4]
db.actors.aggregate(etapas)

// EJERCICIO 18
var query1 = {"cast": {$ne: "Undefined"}}
var fase1 = {$match: query1}
var query2 = {"_id": {"título": "$title", "año": "$year"}, "num_actors": {$sum: 1}}
var fase2 = {$group: query2}
var fase3 = {$sort: {"num_actors": -1}}
var fase4 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4]
db.actors.aggregate(etapas)

// EJERCICIO 19
var query1 = {"cast": {$ne: "Undefined"}}
var fase1 = {$match: query1}
var query2 = {"_id": "$cast", "comienza": {$min: "$year"}, "termina": {$max: "$year"}}
var fase2 = {$group: query2}
var query3 = {"_id": 1, "comienza": 1, "termina": 1, "años": {$subtract: [{$add: ["$termina", 1]}, "$comienza"]}}
var fase3 = {$project: query3}
var fase4 = {$sort: {"años": -1}}
var fase5 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4, fase5]
db.actors.aggregate(etapas)

// EJERCICIO 20
var fase1 = {$unwind: "$genres"}
var query2 = {"_id": 0}
var fase2 = {$project: query2}
var fase3 = {$out: "genres"}
var etapas = [fase1, fase2, fase3]
db.actors.aggregate(etapas)
db.genres.count()

// EJERCICIO 21
var query1 = {"_id": {"año": "$year", "género": "$genres"}, "peliculas": {$addToSet: "$title"}}
var fase1 = {$group: query1}
var query2 = {"_id": 1, "num_películas_dif": {$size: "$peliculas"}}
var fase2 = {$project: query2}
var fase3 = {$sort: {"num_películas_dif": -1}}
var fase4 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4]
db.genres.aggregate(etapas)

// EJERCICIO 22

var query1 = {"cast": {$ne: "Undefined"}}
var fase1 = {$match: query1}
var query2 = {"_id": "$cast", "géneros": {$addToSet: "$genres"},}
var fase2 = {$group: query2}
var query3 = {"_id": 1, "num_genres": {$size: "$géneros"}, "géneros": 1}
var fase3 = {$project: query3}
var fase4 = {$sort: {"num_genres": -1}}
var fase5 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4, fase5]
db.genres.aggregate(etapas)

// Veo que no obtengo el resultado en el mismo orden que el pedido. Por lo que he leído
// veo que el orden lo impone MongoDB. Si quiero definir explícitamente el orden debería usar
// addFields, pero como no se especifica nada en relación con el orden de los resultados, consideraré
// irrelevante esto último.

// EJERCICIO 23
var query1 = {"_id": {"título": "$title", "año": "$year"}, "géneros": {$addToSet: "$genres"}}
var fase1 = {$group: query1}
var query2 = {"num_genres": {$size: "$géneros"}, "géneros": 1}
var fase2 = {$project: query2}
var fase3 = {$sort: {"num_genres": -1}}
var fase4 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4]
db.genres.aggregate(etapas)

// Con respecto al orden del resultado, ídem que en el ejercicio anterior.

// EJERCICIO 24 (LIBRE)

// Listado con la cantidad de películas de cada género ordenadas de mayor a menor
var query1 = {$and: [{"cast": {$ne: ["Undefined"]}}, {"genres": {$ne: ["Undefined"]}}]}
var fase1 = {$match: query1}
var fase2 = {$unwind: "$genres"}
var query3 = {"_id": "$genres", "num_peliculas": {$sum: 1}}
var fase3 = {$group: query3}
var fase4 = {$sort: {"num_peliculas": -1}}
var etapas = [fase1, fase2, fase3, fase4]
db.movies.aggregate(etapas)

// EJERCICIO 25 (LIBRE)

// Cantidad de géneros diferentes y listado de estos para las últimas 5 décadas
var query1 = {$and: [{"cast": {$ne: ["Undefined"]}}, {"genres": {$ne: ["Undefined"]}}]}
var fase1 = {$match: query1}
var fase2 = {$unwind: "$genres"}
var query3 = {"década": {$subtract: ["$year", {$mod: ["$year", 10]}]}, "genres": 1}
var fase3 = {$project: query3}
var query4 = {"_id": "$década", "dif_genres": {$addToSet: "$genres"}}
var fase4 = {$group: query4}
var query5 = {"_id": 1, "num_dif_genres": {$size: "$dif_genres"}, "dif_genres": 1}
var fase5 = {$project: query5}
var fase6 = {$sort: {"_id": -1}}
var fase7 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4, fase5, fase6, fase7]
db.movies.aggregate(etapas)

// EJERCICIO 26 (LIBRE)

// Los cinco géneros con mayor promedio de películas anuales
var query1 = {"genres": {$ne: ["Undefined"]}}
var fase1 = {$match: query1}
var fase2 = {$unwind: "$genres"}
var query3 = {"_id": {"año": "$year", "género": "$genres"}, "películas": {$sum: 1}}
var fase3 = {$group: query3}
var query4 = {"_id": "$_id.género", "total_películas": {$sum: "$películas"},
              "primer_registro": {$min: "$_id.año"}, "último_registro": {$max: "$_id.año"}}
var fase4 = {$group: query4}
var query5 = {"_id": 1, "primer_registro": 1, "último_registro": 1,
              "avg_pelis_anuales": {$round: [{$divide: ["$total_películas",{$add: [{$subtract: ["$último_registro", "$primer_registro"]}, 1]}]},2]}}
var fase5 = {$project: query5}
var fase6 = {$sort: {"avg_pelis_anuales": -1}}
var fase7 = {$limit: 5}
var etapas = [fase1, fase2, fase3, fase4, fase5, fase6, fase7]
db.movies.aggregate(etapas)