# app.py - FastAPI API مع تكامل نموذج PlantVillage
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import uuid
from datetime import datetime
import uvicorn
import logging
from typing import Optional

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tomato Disease Detection API - PlantVillage Model")

# السماح بجميع الأصول
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج
MODEL_PATH = "tomato_model.h5"  # مسار نموذجك المدرب
LABELS = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight", 
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# تحميل النموذج
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# قاعدة بيانات الأمراض بناءً على PlantVillage
DISEASES_DB = {
    "Tomato_Bacterial_spot": {
        "id": "BS-001",
        "name": "البقعة البكتيرية في الطماطم",
        "arabic_name": "البقعة البكتيرية",
        "type": "بكتيري",
        "severity": "متوسط",
        "symptoms": [
            "بقع مائية صغيرة على الأوراق",
            "تحول البقع إلى اللون البني الداكن",
            "هالات صفراء حول البقع",
            "تساقط الأوراق في الحالات الشديدة"
        ],
        "treatment": {
            "immediate": [
                "إزالة الأوراق المصابة وتدميرها",
                "رش مبيد نحاسي (كوبر أوكسيد كلورايد)",
                "تجنب الري العلوي"
            ],
            "biological": [
                "استخدام بكتيريا Bacillus subtilis",
                "رش مستخلص نباتي (القرنفل، النيم)"
            ],
            "prevention": [
                "تعقيم البذور",
                "تناوب المحاصيل",
                "تحسين التهوية"
            ]
        },
        "transmission": "بكتيريا Xanthomonas spp.",
        "risk_level": "medium",
        "prevention_tips": "استخدم بذور معتمدة، تجنب الرطوبة العالية"
    },
    "Tomato_Early_blight": {
        "id": "EB-002",
        "name": "الندوة المبكرة في الطماطم",
        "arabic_name": "الندوة المبكرة",
        "type": "فطري",
        "severity": "متوسط",
        "symptoms": [
            "بقع بنية دائرية على الأوراق القديمة",
            "حلقات متحدة المركز تشبه الهدف",
            "اصفرار الأوراق حول البقع",
            "تساقط الأوراق من الأسفل إلى الأعلى"
        ],
        "treatment": {
            "immediate": [
                "رش مبيد فطري (كلوروثالونيل)",
                "إزالة الأوراق المصابة",
                "تحسين التهوية"
            ],
            "biological": [
                "استخدام Trichoderma harzianum",
                "الرش بمستخلص نباتي"
            ],
            "prevention": [
                "تناوب المحاصيل لمدة 3 سنوات",
                "التسميد المتوازن",
                "التقليم لتحسين التهوية"
            ]
        },
        "transmission": "فطر Alternaria solani",
        "risk_level": "medium"
    },
    "Tomato_Late_blight": {
        "id": "LB-003",
        "name": "الندوة المتأخرة في الطماطم",
        "arabic_name": "الندوة المتأخرة",
        "type": "فطري",
        "severity": "عالي",
        "symptoms": [
            "بقع خضراء باهتة على الأوراق",
            "تحول البقع إلى بني غامق",
            "عفن أبيض على السطح السفلي للأوراق",
            "تلف سريع للنبات"
        ],
        "treatment": {
            "immediate": [
                "رش مبيد نظامي (ميتالاكسيل)",
                "إزالة النباتات المصابة بالكامل",
                "تجنب الري العلوي"
            ],
            "biological": [
                "استخدام Bacillus amyloliquefaciens",
                "مستخلصات نباتية مضادة للفطريات"
            ],
            "prevention": [
                "زراعة أصناف مقاومة",
                "التباعد بين النباتات",
                "الصرف الجيد للمياه"
            ]
        },
        "transmission": "فطر Phytophthora infestans",
        "risk_level": "high"
    },
    "Tomato_Leaf_Mold": {
        "id": "LM-004",
        "name": "عفن الأوراق في الطماطم",
        "arabic_name": "عفن الأوراق",
        "type": "فطري",
        "severity": "متوسط",
        "symptoms": [
            "بقع صفراء على السطح العلوي للأوراق",
            "عفن أرجواني إلى بني على السطح السفلي",
            "تساقط الأوراق",
            "ضعف نمو النبات"
        ],
        "treatment": {
            "immediate": [
                "تحسين التهوية",
                "رش مبيد فطري (أزوكسيستروبين)",
                "إزالة الأوراق المصابة"
            ],
            "biological": [
                "استخدام فطر Trichoderma",
                "تقليل الرطوبة"
            ],
            "prevention": [
                "التحكم في الرطوبة",
                "زراعة أصناف مقاومة",
                "تنظيم درجة الحرارة"
            ]
        },
        "transmission": "فطر Fulvia fulva",
        "risk_level": "medium"
    },
    "Tomato_Septoria_leaf_spot": {
        "id": "SLS-005",
        "name": "بقعة سبتوريا الأوراق",
        "arabic_name": "بقعة سبتوريا",
        "type": "فطري",
        "severity": "متوسط",
        "symptoms": [
            "بقع دائرية صغيرة على الأوراق القديمة",
            "مراكز البقع رمادية مع حواف بنية",
            "نقاط سوداء صغيرة في مركز البقع",
            "اصفرار وتساقط الأوراق"
        ],
        "treatment": {
            "immediate": [
                "رش مبيد فطري (مانكوزيب)",
                "إزالة الأوراق المصابة",
                "تحسين دوران الهواء"
            ],
            "biological": [
                "مستخلصات نباتية مضادة للفطريات",
                "البكتيريا المضادة"
            ],
            "prevention": [
                "تناوب المحاصيل",
                "تنظيف بقايا المحصول",
                "الري بالتنقيط"
            ]
        },
        "transmission": "فطر Septoria lycopersici",
        "risk_level": "medium"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "id": "SM-006",
        "name": "حلم العنكبوت ذو البقعتين",
        "arabic_name": "حلم العنكبوت",
        "type": "آفة حشرية",
        "severity": "عالي",
        "symptoms": [
            "بقع صفراء صغيرة على الأوراق",
            "نسيج عنكبوتي دقيق على السطح السفلي",
            "تجعد الأوراق وجفافها",
            "توقف النمو"
        ],
        "treatment": {
            "immediate": [
                "رش مبيد حشري (أبامكتين)",
                "غسل الأوراق بالماء والصابون",
                "إزالة الأوراق المصابة بشدة"
            ],
            "biological": [
                "إطلاق المفترس Phytoseiulus persimilis",
                "استخدام زيت النيم"
            ],
            "prevention": [
                "الري المنتظم",
                "التهوية الجيدة",
                "المراقبة الدورية"
            ]
        },
        "transmission": "حلم العنكبوت Tetranychus urticae",
        "risk_level": "high"
    },
    "Tomato_Target_Spot": {
        "id": "TS-007",
        "name": "البقعة المستهدفة",
        "arabic_name": "البقعة المستهدفة",
        "type": "فطري",
        "severity": "متوسط",
        "symptoms": [
            "بقع دائرية بنية على الأوراق",
            "حلقات متحدة المركز تشبه الهدف",
            "تساقط الأوراق من الأسفل",
            "بقع على السيقان والثمار"
        ],
        "treatment": {
            "immediate": [
                "رش مبيد فطري (كلوروثالونيل)",
                "إزالة الأجزاء المصابة",
                "تحسين التهوية"
            ],
            "biological": [
                "مستخلصات نباتية",
                "المكافحة الحيوية"
            ],
            "prevention": [
                "التقليم المنتظم",
                "تجنب الرطوبة العالية",
                "النظافة البستانية"
            ]
        },
        "transmission": "فطر Corynespora cassiicola",
        "risk_level": "medium"
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "id": "TYLCV-008",
        "name": "فيروس تجعد الأوراق الصفراء",
        "arabic_name": "فيروس تجعد الأوراق الصفراء",
        "type": "فيروسي",
        "severity": "عالي",
        "symptoms": [
            "تجعد حواف الأوراق إلى الأعلى",
            "اصفرار حواف الأوراق",
            "تقزم النبات",
            "توقف الإثمار"
        ],
        "treatment": {
            "immediate": [
                "إزالة النباتات المصابة وتدميرها",
                "رش مبيد للحشرة الناقلة",
                "استخدام شبكات واقية"
            ],
            "biological": [
                "المكافحة البيولوجية للذباب الأبيض",
                "مصائد لاصقة صفراء"
            ],
            "prevention": [
                "زراعة أصناف مقاومة",
                "استخدام شتلات سليمة",
                "المباعدة بين الزراعات"
            ]
        },
        "transmission": "الذباب الأبيض Bemisia tabaci",
        "risk_level": "high"
    },
    "Tomato_Tomato_mosaic_virus": {
        "id": "TMV-009",
        "name": "فيروس موزاييك الطماطم",
        "arabic_name": "فيروس موزاييك الطماطم",
        "type": "فيروسي",
        "severity": "عالي",
        "symptoms": [
            "تبقع فسيفسائي على الأوراق",
            "تقزم النمو",
            "تشوه الأوراق",
            "انخفاض الإنتاجية"
        ],
        "treatment": {
            "immediate": [
                "إزالة النباتات المصابة",
                "تعقيم الأدوات",
                "غسل اليدين قبل التعامل"
            ],
            "biological": [
                "لا يوجد علاج بيولوجي",
                "الوقاية هي الأساس"
            ],
            "prevention": [
                "استخدام بذور معتمدة",
                "تعقيم التربة",
                "تجنب التدخين في البيوت المحمية"
            ]
        },
        "transmission": "ميكانيكي، التلامس",
        "risk_level": "high"
    },
    "Tomato_healthy": {
        "id": "HEALTHY-000",
        "name": "نبات سليم",
        "arabic_name": "نبات سليم",
        "type": "سليم",
        "severity": "لا يوجد",
        "symptoms": ["لا توجد أعراض مرضية"],
        "treatment": {
            "immediate": ["المتابعة الدورية"],
            "biological": ["الوقاية المستمرة"],
            "prevention": ["الفحص الدوري", "التسميد المتوازن"]
        },
        "transmission": "لا يوجد",
        "risk_level": "none"
    }
}

def preprocess_image(image_bytes, target_size=(224, 224)):
    """معالجة الصورة للإدخال في النموذج"""
    try:
        # فتح الصورة وتحويلها
        image = Image.open(io.BytesIO(image_bytes))
        
        # تحويل إلى RGB إذا كان في وضع آخر
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # تغيير الحجم
        image = image.resize(target_size)
        
        # تحويل إلى مصفوفة numpy
        img_array = np.array(image)
        
        # تطبيع الصورة
        img_array = img_array / 255.0
        
        # إضافة بُعد الدُفعة
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_disease(image_array):
    """التنبؤ بالمرض باستخدام النموذج"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # التنبؤ
        predictions = model.predict(image_array)
        
        # الحصول على أعلى درجة ثقة
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # الحصول على اسم المرض
        predicted_class = LABELS[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    """صفحة الترحيب"""
    return {
        "message": "Welcome to Tomato Disease Detection API",
        "version": "1.0.0",
        "model": "PlantVillage Tomato Diseases",
        "available_diseases": len(LABELS),
        "status": "active"
    }

@app.get("/api/health")
async def health_check():
    """فحص صحة API والنموذج"""
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "available_classes": LABELS if model else [],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict")
async def predict_disease_from_image(
    file: UploadFile = File(...),
    growth_stage: Optional[str] = None,
    cultivation_type: Optional[str] = None
):
    """التنبؤ بالمرض من صورة"""
    
    # التحقق من نوع الملف
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # قراءة الصورة
        contents = await file.read()
        
        # معالجة الصورة
        processed_image = preprocess_image(contents)
        
        # التنبؤ
        disease_class, confidence = predict_disease(processed_image)
        
        # الحصول على معلومات المرض
        disease_info = DISEASES_DB.get(disease_class, DISEASES_DB["Tomato_healthy"])
        
        # إنشاء الرد
        response = {
            "success": True,
            "prediction_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "disease": {
                "class": disease_class,
                "name": disease_info["name"],
                "arabic_name": disease_info["arabic_name"],
                "type": disease_info["type"],
                "severity": disease_info["severity"],
                "confidence": round(confidence, 2)
            },
            "symptoms": disease_info["symptoms"],
            "treatment": disease_info["treatment"],
            "transmission": disease_info["transmission"],
            "risk_level": disease_info["risk_level"],
            "additional_info": {
                "growth_stage": growth_stage,
                "cultivation_type": cultivation_type,
                "image_size": len(contents)
            }
        }
        
        logger.info(f"Prediction successful: {disease_class} ({confidence:.2f}%)")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diseases")
async def get_all_diseases():
    """الحصول على قائمة جميع الأمراض"""
    diseases_list = []
    for class_name in LABELS:
        disease_info = DISEASES_DB.get(class_name, DISEASES_DB["Tomato_healthy"])
        diseases_list.append({
            "class": class_name,
            "name": disease_info["name"],
            "arabic_name": disease_info["arabic_name"],
            "type": disease_info["type"],
            "severity": disease_info["severity"]
        })
    
    return {
        "count": len(diseases_list),
        "diseases": diseases_list
    }

@app.get("/api/disease/{disease_class}")
async def get_disease_details(disease_class: str):
    """الحصول على تفاصيل مرض معين"""
    if disease_class not in DISEASES_DB:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    disease_info = DISEASES_DB[disease_class]
    
    return {
        "disease": disease_info,
        "prevention_tips": disease_info.get("prevention_tips", "")
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
