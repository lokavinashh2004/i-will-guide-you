import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from googleapiclient.discovery import build
import os
import google.generativeai as genai
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random string

class SkillJobVideoRecommender:
    def __init__(self, youtube_api_key, gemini_api_key):
        """
        Initialize the skill and job role video recommender.
        
        Args:
            youtube_api_key (str): YouTube Data API key
            gemini_api_key (str): Gemini API key
        """
        if not youtube_api_key or not gemini_api_key:
            raise ValueError("Both YouTube API key and Gemini API key are required")
        
        # Setup YouTube API
        self.youtube_api_key = youtube_api_key
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        self.video_data = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # Setup Gemini API
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def get_skills_for_job(self, job_role):
        """
        Use Gemini API to get relevant skills for a job role.
        
        Args:
            job_role (str): The job role to find skills for
            
        Returns:
            list: List of skills needed for the job role
        """
        prompt = f"""
        List the top 5-7 technical skills required for a {job_role} position.
        Format the output as a JSON array of strings.
        Example format: ["skill1", "skill2", "skill3"]
        Only include the raw JSON array, no additional text or explanation.
        """
        
        response = self.model.generate_content(prompt)
        
        try:
            # Extract and parse JSON from the response
            skills_text = response.text.strip()
            # Clean up the response to ensure it's valid JSON
            if skills_text.startswith('```json'):
                skills_text = skills_text.replace('```json', '').replace('```', '')
            skills_list = json.loads(skills_text)
            return skills_list
        except json.JSONDecodeError:
            # Fallback parsing for non-standard responses
            skills_text = response.text.strip()
            # Try to extract anything that looks like a list
            if '[' in skills_text and ']' in skills_text:
                list_part = skills_text[skills_text.find('['): skills_text.rfind(']') + 1]
                try:
                    return json.loads(list_part)
                except:
                    pass
            
            # If all else fails, parse manually by looking for patterns
            skills = []
            for line in skills_text.split('\n'):
                if '-' in line or ':' in line or '.' in line:
                    # Extract the skill name, removing numbers, dashes, etc.
                    skill = line.split('-')[-1].split(':')[-1].split('.')[-1].strip()
                    if skill and len(skill) > 2:  # Basic validation
                        skills.append(skill)
            
            return skills if skills else ["Python programming", "Data analysis", "Machine learning"]  # Default fallback
        
    def search_videos(self, skill, max_results=20):
        """
        Search YouTube for videos related to a skill.
        
        Args:
            skill (str): The skill to search for
            max_results (int): Maximum number of results to return
        
        Returns:
            pandas.DataFrame: DataFrame containing video information
        """
        search_query = f"learn {skill} tutorial"
        
        try:
            # Initial search to find relevant videos
            search_response = self.youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=max_results,
                type='video',
                relevanceLanguage='en',
                order='relevance'
            ).execute()
            
            # Check if we have results
            if 'items' not in search_response or len(search_response['items']) == 0:
                print(f"No videos found for '{skill}'")
                self.video_data = pd.DataFrame()
                return self.video_data
            
            # Extract video IDs
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Get video statistics
            videos_response = self.youtube.videos().list(
                id=','.join(video_ids),
                part='snippet,contentDetails,statistics'
            ).execute()
            
            # Prepare data for DataFrame
            video_data = []
            for item in videos_response['items']:
                # Handle missing statistics with safe get
                statistics = item.get('statistics', {})
                video_info = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'view_count': int(statistics.get('viewCount', 0)),
                    'like_count': int(statistics.get('likeCount', 0)),
                    'comment_count': int(statistics.get('commentCount', 0)),
                    'duration': item['contentDetails'].get('duration', 'PT0M0S')
                }
                video_data.append(video_info)
                
            self.video_data = pd.DataFrame(video_data)
            return self.video_data
            
        except Exception as e:
            print(f"Error searching for videos: {e}")
            self.video_data = pd.DataFrame()
            return self.video_data
    
    def preprocess_data(self):
        """
        Preprocess the video data for recommendation.
        """
        if self.video_data is None or self.video_data.empty:
            # Instead of raising error, return early
            print("No video data available for preprocessing")
            return
        
        try:
            # Create a text feature combining title and description
            self.video_data['content'] = self.video_data['title'] + ' ' + self.video_data['description']
            
            # Calculate like ratio safely
            self.video_data['like_ratio'] = self.video_data['like_count'] / (self.video_data['view_count'] + 1)
            
            # Parse published dates safely
            try:
                self.video_data['published_date'] = pd.to_datetime(self.video_data['published_at'])
                self.video_data['recency_score'] = 0.5  # Default value
                
                # Only calculate recency if we have more than one date
                if len(self.video_data['published_date'].unique()) > 1:
                    max_date = self.video_data['published_date'].max()
                    min_date = self.video_data['published_date'].min()
                    date_range = max_date - min_date
                    
                    if date_range.total_seconds() > 0:
                        # Calculate recency as normalized position in date range
                        self.video_data['recency_score'] = (
                            (self.video_data['published_date'] - min_date) / date_range
                        ).astype(float)
                    
            except Exception as e:
                print(f"Error processing dates: {e}")
                self.video_data['recency_score'] = 0.5  # Default if date processing fails
            
            # Calculate engagement score with safe fallbacks
            self.video_data['engagement_score'] = (
                self.video_data['view_count'] * 0.4 + 
                self.video_data['like_count'] * 0.4 + 
                self.video_data['comment_count'] * 0.1 +
                self.video_data['like_ratio'] * 0.3 * self.video_data['view_count'] +
                self.video_data['recency_score'] * 0.2 * self.video_data['view_count']
            )
            
            # Normalize engagement score
            max_engagement = self.video_data['engagement_score'].max()
            if max_engagement > 0:
                self.video_data['engagement_score'] = self.video_data['engagement_score'] / max_engagement
            else:
                self.video_data['engagement_score'] = 0.5  # Default if normalization fails
                
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.video_data['content'])
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Set default values if preprocessing fails
            self.video_data['relevance_score'] = 0.5
            self.video_data['engagement_score'] = 0.5
            self.video_data['final_score'] = 0.5
            
            # Create a simple TF-IDF matrix if needed
            if 'content' not in self.video_data.columns:
                self.video_data['content'] = self.video_data['title'] 
            self.tfidf_matrix = self.vectorizer.fit_transform(self.video_data['content'])
        
    def get_recommendations(self, skill_query, top_n=3):
        """
        Get video recommendations for a specific skill query.
        
        Args:
            skill_query (str): The skill to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            pandas.DataFrame: Top recommended videos
        """
        if self.video_data is None or self.video_data.empty:
            print(f"No videos available to recommend for {skill_query}")
            return pd.DataFrame()
            
        try:
            if self.tfidf_matrix is None:
                self.preprocess_data()
                
            if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
                print(f"No content available for recommendations for {skill_query}")
                return self.video_data.head(min(top_n, len(self.video_data)))
                
            # Transform the query
            query_vec = self.vectorizer.transform([skill_query])
            
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Add similarity scores to DataFrame
            self.video_data['relevance_score'] = cosine_similarities
            
            # Calculate final score combining relevance and engagement
            if 'engagement_score' not in self.video_data.columns:
                self.video_data['engagement_score'] = 0.5  # Default if missing
                
            self.video_data['final_score'] = (
                0.6 * self.video_data['relevance_score'] + 
                0.4 * self.video_data['engagement_score']
            )
            
            # Get top recommendations
            recommendations = self.video_data.sort_values(
                by='final_score', ascending=False
            ).head(top_n)
            
            # Convert the DataFrame to a dictionary for JSON serialization
            recommendations_dict = recommendations.to_dict(orient='records')
            
            return recommendations_dict
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            # Return top videos by view count as fallback
            if 'view_count' in self.video_data.columns:
                fallback = self.video_data.sort_values(by='view_count', ascending=False).head(top_n)
                return fallback.to_dict(orient='records')
            else:
                return self.video_data.head(min(top_n, len(self.video_data))).to_dict(orient='records')


# Create an instance of the recommender
def get_recommender():
    youtube_api_key = os.environ.get('YOUTUBE_API_KEY', "AIzaSyBatHQaaWZmBxh_sz0z46m5NTI4dc5KRcc")
    gemini_api_key = os.environ.get('GEMINI_API_KEY', "AIzaSyDPmukhY7Ejs9TEwaRyxtCMiTZVAsJC2dk")
    
    return SkillJobVideoRecommender(youtube_api_key, gemini_api_key)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skill_search', methods=['POST'])
def skill_search():
    skill = request.form.get('skill')
    
    if not skill:
        return jsonify({'error': 'Please enter a skill'}), 400
    
    try:
        recommender = get_recommender()
        recommender.search_videos(skill)
        recommendations = recommender.get_recommendations(skill)
        
        if not recommendations:
            return jsonify({'error': f"Couldn't find recommendations for {skill}"}), 404
            
        return jsonify({'recommendations': recommendations, 'skill': skill})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job_search', methods=['POST'])
def job_search():
    job_role = request.form.get('job_role')
    
    if not job_role:
        return jsonify({'error': 'Please enter a job role'}), 400
    
    try:
        recommender = get_recommender()
        skills = recommender.get_skills_for_job(job_role)
        
        if not skills:
            return jsonify({'error': f"Couldn't identify skills for {job_role}"}), 404
        
        all_recommendations = {}
        for skill in skills:
            recommender.search_videos(skill)
            skill_recommendations = recommender.get_recommendations(skill)
            if skill_recommendations:
                all_recommendations[skill] = skill_recommendations
        
        response_data = {
            'job_role': job_role,
            'skills': skills,
            'recommendations': all_recommendations
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)