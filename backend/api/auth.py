"""
OralGuard AI — Authentication API Routes

Endpoints:
  POST /auth/register — Register new user
  POST /auth/login    — Login and get JWT token
  GET  /auth/me       — Get current user profile
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from passlib.context import CryptContext
from jose import JWTError, jwt

from database import get_db
from config import settings
from models.db_models import User, UserRole
from models.schemas import UserCreate, UserResponse, TokenResponse, LoginRequest


router = APIRouter(prefix="/auth", tags=["Authentication"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account."""
    # Check existing user
    filters = []
    if user_data.email:
        filters.append(User.email == user_data.email)
    if user_data.phone:
        filters.append(User.phone == user_data.phone)

    if filters:
        result = await db.execute(select(User).where(or_(*filters)))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="User with this email or phone already exists",
            )

    # Create user
    user = User(
        email=user_data.email,
        phone=user_data.phone,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        role=UserRole(user_data.role),
        age=user_data.age,
        gender=user_data.gender,
        state=user_data.state,
        district=user_data.district,
    )
    db.add(user)
    await db.flush()

    token = create_access_token({"sub": user.id, "role": user.role.value})

    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Login with email/phone and password."""
    filters = []
    if credentials.email:
        filters.append(User.email == credentials.email)
    if credentials.phone:
        filters.append(User.phone == credentials.phone)

    if not filters:
        raise HTTPException(status_code=400, detail="Email or phone required")

    result = await db.execute(select(User).where(or_(*filters)))
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token({"sub": user.id, "role": user.role.value})

    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    db: AsyncSession = Depends(get_db),
    # In production: token = Depends(oauth2_scheme)
):
    """Get current authenticated user profile."""
    # Placeholder — in production, decode JWT and fetch user
    raise HTTPException(
        status_code=501,
        detail="Authentication middleware not yet configured. "
               "Pass JWT token in Authorization header.",
    )
