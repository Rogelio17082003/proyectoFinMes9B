from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS  # Importar CORS

app = Flask(__name__)

# Configurar CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Permitir todos los orígenes

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos
ordinal_encoder = joblib.load('modelo_ordinalEncoder.pkl')
scaler = joblib.load('modelo_Scaler.pkl')
modelRF = joblib.load('modelo_RandomForest.pkl')

app.logger.debug('4 modelos cargados correctamente.')
@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            # Procesar el DataFrame y realizar la predicción
            df = df['Sexo']
            
            return jsonify({'message': df.to_string(index=False)})  # Devuelve el DataFrame con predicciones
        return jsonify({'message': 'Archivo no válido'})
    except Exception as e:
        return jsonify({'message': f'Error al procesar el archivo: {str(e)}'}), 500



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Carrera = request.form['Carrera']
        Escuela = request.form['Escuela']
        Municipio = request.form['Municipio']
        Estado = request.form['Estado']
        Sexo = request.form['Sexo']
        Estado_Civil = request.form['Estado_Civil']
        calFin = float(request.form['calFin'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[Carrera, Escuela, Municipio, Estado, Sexo, Estado_Civil, calFin]], 
                               columns=['Carrera Escogida', 'Escuela de Origen', 'Municipio', 'Estado', 'Sexo', 'Estado Civil', 'calFin'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Seleccionar las columnas categóricas a convertir a numérico
        categorical_columns = ['Carrera Escogida', 'Escuela de Origen',
                       'Municipio', 'Estado', 'Sexo', 'Estado Civil', 'calFin']
        # Transformar las columnas categóricas a numérico
        data_df[categorical_columns] = ordinal_encoder.transform(data_df[categorical_columns])
        app.logger.debug(f'Datos transformados a numérico: {data_df}')

        # Escalar los datos
        scaler_df = scaler.transform(data_df)
        scaler_df = pd.DataFrame(scaler_df, columns=data_df.columns)
        app.logger.debug(f'DataFrame escalado: {scaler_df}')

        # Realizar la predicción
        prediction = modelRF.predict(scaler_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Convertir la predicción a un tipo de datos serializable (int)
        prediction_serializable = int(prediction[0])

        # Mapear la predicción a una categoría, ajusta según las categorías de tu modelo
        if prediction_serializable == 0:
            category = "Reprovar"
        elif prediction_serializable == 1:
            category = "Pasar"
        else:
            category = "Unknown"

        # Devolver la predicción como respuesta JSON
        return jsonify({'categoria': category})
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
