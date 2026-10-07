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
try:
    # 1. Try PyJWT (modern standard: pip install pyjwt)
    import jwt
    from jwt.exceptions import PyJWTError as JWTError
except (ImportError, SyntaxError):
    try:
        # 2. Try python-jose (if installed without conflicting jose.py)
        from jose import JWTError, jwt  # type: ignore
    except (ImportError, SyntaxError):
        # 3. Built-in zero-dependency HMAC-SHA256 JWT fallback
        import hmac
        import hashlib
        import base64
        import json

        class JWTError(Exception):
            pass

        class SimpleJWT:
            @staticmethod
            def encode(claims: dict, key: str, algorithm: str = "HS256") -> str:
                header = {"alg": "HS256", "typ": "JWT"}
                h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
                c_b64 = base64.urlsafe_b64encode(json.dumps(claims, default=str).encode()).decode().rstrip("=")
                sig = hmac.new(key.encode(), f"{h_b64}.{c_b64}".encode(), hashlib.sha256).digest()
                s_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
                return f"{h_b64}.{c_b64}.{s_b64}"

            @staticmethod
            def decode(token: str, key: str, algorithms: list = None) -> dict:
                parts = token.split(".")
                if len(parts) != 3:
                    raise JWTError("Invalid token format")
                h_b64, c_b64, s_b64 = parts
                expected_sig = hmac.new(key.encode(), f"{h_b64}.{c_b64}".encode(), hashlib.sha256).digest()
                sig_pad = s_b64 + "=" * (-len(s_b64) % 4)
                if not hmac.compare_digest(base64.urlsafe_b64decode(sig_pad), expected_sig):
                    raise JWTError("Signature verification failed")
                c_pad = c_b64 + "=" * (-len(c_b64) % 4)
                return json.loads(base64.urlsafe_b64decode(c_pad).decode())

        jwt = SimpleJWT()

from database import get_db
from config import settings
from models.db_models import User, UserRole
from models.schemas import UserCreate, UserResponse, TokenResponse, LoginRequest

router = APIRouter(prefix="/auth", tags=["Authentication"])

try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)
except Exception:
    import hashlib
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    def verify_password(plain: str, hashed: str) -> bool:
        return hashlib.sha256(plain.encode()).hexdigest() == hashed


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
