from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
st.set_page_config(page_title='Catalyst Consulting | Analysts & Engineers', page_icon='🔬', layout='wide')

@st.cache_data
def load_data():
    data = pd.read_csv(RESULTS / 'error_analysis_full.csv')
    with open(RESULTS / 'vader_full_metrics.json', encoding='utf-8') as file: vader = json.load(file)
    with open(RESULTS / 'transformer_full_metrics.json', encoding='utf-8') as file: roberta = json.load(file)
    data['text'] = data['text'].fillna('').astype(str)
    data['models_agree'] = data['models_agree'].astype(str).str.strip().str.lower().eq('true')
    return data, vader, roberta

st.caption('CATALYST CONSULTING  /  TECHNICAL ANALYSIS')
st.title('Analysts & Engineers')
if st.button('← Executive overview'):
    st.switch_page('app.py')
try:
    df, vader, roberta = load_data()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f'Unable to load analysis files: {exc}')
    st.stop()
st.info('The reference sentiment labels are derived from star ratings, not human-annotated text sentiment.')

st.header('Full-dataset model comparison')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Reviews', f'{len(df):,}')
c2.metric('VADER rating agreement', f"{vader['accuracy']:.2%}")
c3.metric('RoBERTa rating agreement', f"{roberta['accuracy']:.2%}")
c4.metric('Disagreements', f"{(~df['models_agree']).sum():,}")

labels = ['negative','neutral','positive']
metrics = ['Rating agreement','Macro F1','Negative recall','Neutral recall','Positive recall']
def metric_values(obj):
    report = obj['classification_report']
    return [obj['accuracy'],report['macro avg']['f1-score']] + [report[label]['recall'] for label in labels]
metric_frame = pd.DataFrame({'Metric':metrics, 'VADER':metric_values(vader), 'RoBERTa':metric_values(roberta)})
st.subheader('Performance metrics')
st.bar_chart(metric_frame.set_index('Metric'))
st.dataframe(metric_frame.style.format({'VADER':'{:.1%}','RoBERTa':'{:.1%}'}), hide_index=True, use_container_width=True)

st.subheader('Confusion matrices')
col1, col2 = st.columns(2)
for col, title, data in [(col1,'VADER',vader), (col2,'RoBERTa',roberta)]:
    with col:
        st.markdown('**'+title+'**')
        st.dataframe(pd.DataFrame(data['confusion_matrix'], index=labels, columns=labels), use_container_width=True)
st.caption('Rows = rating-derived sentiment; columns = model predictions.')

st.header('Disagreement diagnostics')
if 'comparison_category' in df:
    st.bar_chart(df['comparison_category'].value_counts())
flags = [column for column in ['has_update_marker','has_html_markup','has_contrast_marker','predictions_opposite'] if column in df.columns]
if flags:
    st.bar_chart(pd.DataFrame({'Indicator':flags,'Reviews':[int(df[f].astype(str).str.lower().eq('true').sum()) for f in flags]}).set_index('Indicator'))
st.caption('Diagnostic flags indicate candidates for investigation and do not establish error causes.')

st.header('Review explorer')
a,b,c = st.columns(3)
with a:
    category = st.selectbox('Comparison outcome', ['All'] + sorted(df['comparison_category'].dropna().unique().tolist()))
with b:
    rating = st.selectbox('Rating-derived sentiment', ['All','negative','neutral','positive'])
with c:
    search = st.text_input('Search full review text')
filtered = df
if category != 'All': filtered = filtered[filtered.comparison_category.eq(category)]
if rating != 'All': filtered = filtered[filtered.rating_sentiment.eq(rating)]
if search: filtered = filtered[filtered.text.str.contains(search, case=False, regex=False, na=False)]
columns = [c for c in ['review_id','rating_sentiment','vader_prediction','transformer_prediction','comparison_category','review_length_words','text'] if c in df.columns]
st.write(f'{len(filtered):,} matching reviews')
st.dataframe(filtered[columns], hide_index=True, use_container_width=True)
st.download_button('Download filtered reviews', filtered.to_csv(index=False).encode('utf-8'), 'catalyst_filtered_reviews.csv','text/csv')
