from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

# تحميل النموذج (تأكد من تدريبه باستخدام Pipeline و N-grams)
try:
    model = pickle.load(open('unified_spam_model.pkl', 'rb'))
except:
    print("⚠️ خطأ: تأكد من وجود ملف unified_spam_model.pkl بجانب السيرفر")

def clean_arabic_english(text):
    text = str(text).lower()
    # توحيد الحروف العربية (أ، إ، آ -> ا) و (ة -> ه)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    # إزالة الرموز والروابط
    text = re.sub(r'http\S+|www\S+|[^\u0600-\u06ff\sa-zA-Z]', ' ', text)
    return ' '.join(text.split())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # 1. تنظيف النص قبل التوقع
        cleaned_msg = clean_arabic_english(message)
        
        # 2. التوقع باستخدام الموديل
        # الموديل سيعيد 'spam' أو 'ham'
        prediction = model.predict([cleaned_msg])[0]
        
        return jsonify({
            'status': 'success',
            'result': prediction.lower() # سيرسل 'spam' لتطبيق فلاتر
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # تأكد من كتابة الـ IP الصحيح هنا أو اتركها 0.0.0.0
    app.run(host='0.0.0.0', port=5000)