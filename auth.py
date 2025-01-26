from fastapi import FastAPI, Depends
from pydantic import BaseModel
import grpc
import auth_pb2
import auth_pb2_grpc

app = FastAPI()

# Define the gRPC client
def get_auth_client():
    channel = grpc.insecure_channel('localhost:44044')  # Replace with the actual address of your gRPC server
    return auth_pb2_grpc.AuthStub(channel)

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str
    app_id: int

class IsAdminRequest(BaseModel):
    user_id: int

@app.post("/register")
def register(request: RegisterRequest, auth_client: auth_pb2_grpc.AuthStub = Depends(get_auth_client)):
    grpc_request = auth_pb2.RegisterRequest(email=request.email, password=request.password)
    response = auth_client.Register(grpc_request)
    return {"user_id": response.user_id}

@app.post("/login")
def login(request: LoginRequest, auth_client: auth_pb2_grpc.AuthStub = Depends(get_auth_client)):
    grpc_request = auth_pb2.LoginRequest(email=request.email, password=request.password, app_id=request.app_id)
    response = auth_client.Login(grpc_request)
    return {"token": response.token}

@app.post("/is_admin")
def is_admin(request: IsAdminRequest, auth_client: auth_pb2_grpc.AuthStub = Depends(get_auth_client)):
    grpc_request = auth_pb2.isAdminRequest(user_id=request.user_id)
    response = auth_client.isAdmin(grpc_request)
    return {"is_admin": response.is_admin}
