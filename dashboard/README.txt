Catalyst Consulting - Executive & Technical Dashboard

Copy dashboard/app.py to the existing project's dashboard/app.py.
Create dashboard/pages/ and place 1_Analysts_and_Engineers.py inside it.
Keep your existing results/ folder with the generated CSV and JSON files.

From an activated Windows CMD in the project root:
    python -m pip install streamlit pandas scikit-learn
    python -m streamlit run dashboard\app.py

The main page uses NMF topic modeling on negative-rated reviews and draws frequency-sized red ovals.
The Analysts & Engineers button switches to the technical page.
This is an exploratory tool: topic labels are generated from keywords and must be validated.
