from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
from geopy.distance import geodesic
from datetime import datetime, date
import math

app = FastAPI()

# Load data from JSON file
def load_data():
    with open('data.json', 'r') as file:
        return json.load(file)

# Models
class User(BaseModel):
    id: str
    userName: str
    name: str
    role: str
    interestedIn: Optional[str] = None
    gender: Optional[str] = None
    dob: Optional[str] = None
    images: List[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius: Optional[float] = None
    address: Optional[str] = None
    bio: Optional[str] = None
    profession: Optional[str] = None
    language: Optional[str] = None

class DataContent(BaseModel):
    myData: User
    usersData: List[User]

class InputData(BaseModel):
    success: bool
    statusCode: int
    message: str
    data: DataContent

class MatchResponse(BaseModel):
    success: bool
    statusCode: int
    message: str
    data: List[User]

def calculate_age(dob_str: str) -> int:
    if not dob_str:
        return 0
    dob = datetime.strptime(dob_str.split('T')[0], '%Y-%m-%d').date()
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if None in [lat1, lon1, lat2, lon2]:
        return float('inf')
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def is_gender_match(user: User, potential_match: User) -> bool:
    if not user.interestedIn or not potential_match.gender:
        return True
    
    if user.interestedIn == "BOTH":
        return True
    elif user.interestedIn == "GIRLS" and potential_match.gender == "FEMALE":
        return True
    elif user.interestedIn == "BOYS" and potential_match.gender == "MALE":
        return True
    return False

def is_age_compatible(user_age: int, match_age: int) -> bool:
    if user_age == 0 or match_age == 0:
        return True
    age_diff = abs(user_age - match_age)
    return age_diff <= 10  # Adjust this threshold as needed

def calculate_similarity(user1: User, user2: User) -> float:
    """Calculate similarity percentage between two users based on various attributes"""
    # Fields to compare (excluding images, id, userName, etc.)
    comparable_fields = [
        ('role', 1),             # Weight 1
        ('gender', 1),           # Weight 1
        ('interestedIn', 1),     # Weight 1
        ('profession', 3),       # Weight 3
        ('bio', 2),              # Weight 2
        ('language', 2),         # Weight 2
        ('address', 1)           # Weight 1
    ]
    
    total_weight = sum(weight for _, weight in comparable_fields)
    matched_weight = 0
    
    # Compare each field
    for field, weight in comparable_fields:
        val1 = getattr(user1, field)
        val2 = getattr(user2, field)
        
        # Skip if either value is None
        if val1 is None or val2 is None:
            total_weight -= weight
            continue
            
        # For language field, check for at least one common language
        if field == 'language' and val1 and val2:
            lang1 = [lang.strip().lower() for lang in val1.split(',')]
            lang2 = [lang.strip().lower() for lang in val2.split(',')]
            if any(l in lang2 for l in lang1):
                matched_weight += weight
        # For other fields, check for exact match
        elif val1 == val2:
            matched_weight += weight
    
    # Age similarity
    age1 = calculate_age(user1.dob) if user1.dob else 0
    age2 = calculate_age(user2.dob) if user2.dob else 0
    if age1 > 0 and age2 > 0:
        age_diff = abs(age1 - age2)
        if age_diff <= 3:
            matched_weight += 2
        elif age_diff <= 7:
            matched_weight += 1
    
    # Calculate similarity percentage
    if total_weight == 0:
        return 0
    
    similarity = (matched_weight / total_weight) * 100
    return similarity

@app.post("/matches", response_model=MatchResponse)
async def get_matches(input_data: InputData):
    # Get the main user's data
    current_user = input_data.data.myData
    
    # Find matches from usersData with similarity calculation
    match_candidates = []
    
    for potential_match in input_data.data.usersData:
        # Skip if it's the same user
        if potential_match.id == current_user.id:
            continue
            
        # Check gender compatibility
        if not is_gender_match(current_user, potential_match):
            continue
            
        # Check distance (within user's radius if specified, otherwise default to 50km)
        distance = calculate_distance(
            current_user.latitude, current_user.longitude,
            potential_match.latitude, potential_match.longitude
        )
        
        radius = current_user.radius or 50  # Default 50km radius
        if distance <= radius:
            # Calculate similarity score for ranking
            similarity = calculate_similarity(current_user, potential_match)
            match_candidates.append((potential_match, similarity))
    
    # Sort matches by similarity score (highest first)
    match_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the user objects for response
    matches = [match for match, _ in match_candidates]
    
    return {
        "success": True,
        "statusCode": 200,
        "message": "Matches retrieved successfully",
        "data": matches
    }

