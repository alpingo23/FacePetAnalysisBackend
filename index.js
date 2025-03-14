const express = require('express');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const { Canvas, Image, ImageData } = canvas;
const fileUpload = require('express-fileupload');
const path = require('path');

const app = express();
app.use(fileUpload());

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Model dosyalarının yolu
const MODEL_PATH = path.join(__dirname, 'models');

// Modelleri yükleme
async function loadModels() {
  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH),
      faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH),
      faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH),
      faceapi.nets.faceExpressionNet.loadFromDisk(MODEL_PATH),
      faceapi.nets.ageGenderNet.loadFromDisk(MODEL_PATH),
    ]);
    console.log('Face-api.js models loaded.');
  } catch (err) {
    console.error('Error loading models:', err);
    throw err;
  }
}

// Sunucuyu başlatmadan önce modelleri yükle
loadModels().then(() => {
  app.listen(5001, () => {
    console.log('Face analysis server running on port 5001');
  });
}).catch(err => {
  console.error('Failed to start server due to model loading error:', err);
});

app.post('/predict_face', async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).json({ error: 'No image provided' });
  }

  const file = req.files.image;

  try {
    // Görüntüyü canvas'a yükle
    const img = await canvas.loadImage(file.data);
    const origWidth = img.width;
    const origHeight = img.height;

    // Görüntüyü analiz et (orijinal boyutlarda)
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender();

    if (detections.length === 0) {
      return res.status(400).json({ error: 'No face detected' });
    }

    // Sonuç objesini oluştur
    const result = detections.map(d => {
      // Koordinatları orijinal boyutlara göre döndür (face-api.js zaten orijinal boyutları kullanır)
      const box = {
        x: d.detection.box.x,
        y: d.detection.box.y,
        width: d.detection.box.width,
        height: d.detection.box.height,
      };
      const landmarks = d.landmarks.positions.map(position => ({
        x: position.x,
        y: position.y,
      }));

      return {
        age: Math.round(d.age),
        gender: d.gender,
        genderProbability: d.genderProbability,
        expressions: d.expressions,
        detection: { box },
        landmarks,
        imageDimensions: { width: origWidth, height: origHeight }, // Orijinal boyutları ekle
      };
    });

    // Loglama: Backend'den gönderilen boyutları ve koordinatları kontrol et
    console.log('Backend Image Dimensions:', result[0].imageDimensions);
    console.log('Backend Detection Box:', result[0].detection.box);
    console.log('Backend Landmarks Sample:', result[0].landmarks.slice(0, 5)); // İlk 5 landmark

    res.json(result[0]); // İlk yüzü döndür
  } catch (err) {
    console.error('Face analysis error:', err);
    res.status(500).json({ error: 'Face analysis failed', details: err.message });
  }
});

module.exports = app;