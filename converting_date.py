import pandas as pd

# قراءة الملف وتجاهل أي صف فيه "Time" أو قيم غير صالحة
df = pd.read_csv("ai_tweets.csv")

# التأكد إن العمود اسمه صح (لو كان 'Time' بدل 'created_at' غير الاسم هنا)
if 'Time' in df.columns:
    df.rename(columns={'Time': 'created_at'}, inplace=True)

# إزالة الصفوف اللي فيها قيم غير صالحة (زي "Time" أو NaN)
df = df[df['created_at'].str.contains(r'\d{4}-\d{2}-\d{2}', na=False)]

# تحويل النصوص إلى تواريخ بشكل تلقائي
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# شيل أي صف فشل التحويل فيه
df = df.dropna(subset=['created_at'])

df['created_at'] = df['created_at']

# حوّله للشكل اللي هينفع في الفيجوالايزيشن
df['created_at'] = df['created_at'].dt.strftime('%m/%d/%Y')

# احفظ الملف الجديد
df.to_csv("ai_tweets.csv", index=False, encoding='utf-8')

print("✅ التواريخ اتعدلت وتخزنت في 'ai_tweets_cleaned.csv'")
