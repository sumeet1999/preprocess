import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import dateparser
import ast
from datetime import datetime

class Preprocessor:
    def __init__(self):
        """Initialize the Preprocessor class with NLTK resources and current date."""
        # Download necessary NLTK data
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.current_date = datetime(2025, 2, 24)  # Set to February 24, 2025

    def clean_simple_text(self, text):
        """Clean text by removing HTML tags, special characters (except #, _, -), and extra whitespace."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s#_-]', '', text)  # Remove special chars except #, _, -
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def preprocess_text(self, text):
        """Preprocess text by cleaning, tokenizing, removing stop words, and lemmatizing."""
        text = self.clean_simple_text(text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def parse_list_str(self, list_str):
        """Parse a string that represents a list or a newline-separated string into a list."""
        if pd.isna(list_str):
            return []
        if isinstance(list_str, str):
            if list_str.startswith('['):
                try:
                    return ast.literal_eval(list_str)
                except:
                    return []
            else:
                return list_str.split('\n')
        return []

    def parse_date(self, date_str):
        """Parse a date string into a datetime object, handling 'Till Date' as the current date."""
        if pd.isna(date_str) or date_str.lower() == 'till date':
            return self.current_date if date_str.lower() == 'till date' else None
        try:
            return dateparser.parse(date_str, settings={'RETURN_AS_TIMEZONE_AWARE': False})
        except:
            return None

    def preprocess_job_description(self, job_description):
        """Preprocess the job description text."""
        return self.preprocess_text(job_description)

    def preprocess_candidate_profile(self, row):
        """Preprocess a single candidate profile (row) from the DataFrame."""
        profile = {}
        
        # Skills
        skills = self.parse_list_str(row['skills'])
        profile['skills'] = [self.clean_simple_text(skill) for skill in skills]
        
        # Experience
        positions = self.parse_list_str(row['positions'])
        start_dates = self.parse_list_str(row['start_dates'])
        end_dates = self.parse_list_str(row['end_dates'])
        responsibilities = self.parse_list_str(row['responsibilities.1'])  # Assuming this holds descriptions
        
        num_experiences = len(positions)
        experience = []
        for i in range(num_experiences):
            exp = {}
            exp['title'] = self.clean_simple_text(positions[i]) if i < len(positions) else ""
            exp['description'] = self.preprocess_text(responsibilities[i]) if i < len(responsibilities) else ""
            exp['start_date'] = self.parse_date(start_dates[i]) if i < len(start_dates) else None
            exp['end_date'] = self.parse_date(end_dates[i]) if i < len(end_dates) else None
            experience.append(exp)
        profile['experience'] = experience
        
        # Education
        institutions = self.parse_list_str(row['educational_institution_name'])
        degrees = self.parse_list_str(row['degree_names'])
        education = []
        for inst, deg in zip(institutions, degrees):
            education.append({
                'degree': self.clean_simple_text(deg),
                'institution': self.clean_simple_text(inst)
            })
        profile['education'] = education
        
        # Summary
        profile['summary'] = self.preprocess_text(row['career_objective'])
        
        # Certifications
        cert_skills = self.parse_list_str(row['certification_skills'])
        profile['certifications'] = [self.clean_simple_text(cert) for cert in cert_skills]
        
        return profile

    def preprocess_candidate_profiles(self, df):
        """Preprocess all candidate profiles in the DataFrame."""
        return df.apply(self.preprocess_candidate_profile, axis=1).tolist()
    


preprocessor = Preprocessor()

job_desc = "Seeking a Big Data Engineer with experience in Hadoop and Spark."
processed_job = preprocessor.preprocess_job_description(job_desc)


df = pd.read_csv("resume_data.csv", na_values=['NA', 'N/A', '--'])
processed_profiles = df.apply(preprocessor.preprocess_candidate_profile, axis=1)

print("\nPreprocessed Candidate Profiles:")
for i, profile in enumerate(processed_profiles):
    print(f"Candidate {i + 1}:")
    print(profile)