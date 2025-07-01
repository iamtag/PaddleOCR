// socket_utils.cpp
#include <include/socket_utils.h>
#include <iostream>
#include <cstring>

#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
//#include <unistd.h>
//#include <netinet/in.h>
//#include <sys/socket.h>
//#include <arpa/inet.h>

#include <include/utility.h>

SocketServer::SocketServer(int port) : port_(port), server_fd_(INVALID_SOCKET), client_fd_(INVALID_SOCKET) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        perror("WSAStartup failed");
        exit(EXIT_FAILURE);
    }
}

SocketServer::~SocketServer() {
    if (client_fd_ != INVALID_SOCKET) closesocket(client_fd_);
    if (server_fd_ != INVALID_SOCKET) closesocket(server_fd_);
}

void SocketServer::start() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == INVALID_SOCKET) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd_, 3) == SOCKET_ERROR) {
        perror("listen failed");
        exit(EXIT_FAILURE);
    }

    PaddleOCR::Utility::log_with_timestamp("[INFO] OCR service started. Listening on port ") << port_ << std::endl;
}

std::string SocketServer::receive() {
    sockaddr_in client_addr;
    int client_len = sizeof(client_addr);
    client_fd_ = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd_ == INVALID_SOCKET) {
        perror("accept failed");
        return "";
    }

    char buffer[8192] = {0};
    int bytes_read = recv(client_fd_, buffer, sizeof(buffer), 0);
    if (bytes_read <= 0) {
        perror("read failed");
        return "";
    }
    return std::string(buffer, bytes_read);
}

void SocketServer::send(const std::string& data) {
    if (client_fd_ == INVALID_SOCKET) return;
    if (!data.empty()) {
		size_t total_sent = 0;
		size_t data_size = data.size();
		while (total_sent < data_size) {
			int sent = ::send(client_fd_, data.c_str() + total_sent, data_size - total_sent, 0);
			if (sent == SOCKET_ERROR) {
				perror("send failed");
				return;
			}
			total_sent += sent;
		}
    }
    closesocket(client_fd_);
    client_fd_ = INVALID_SOCKET;
}



SocketClient::SocketClient(int port)
	: port_(port), client_fd_(INVALID_SOCKET) {
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		perror("WSAStartup failed");
		exit(EXIT_FAILURE);
	}
}

SocketClient::~SocketClient() {
	if (client_fd_ != INVALID_SOCKET) closesocket(client_fd_);
	WSACleanup();
}

void SocketClient::connect() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd_ == INVALID_SOCKET) {
        std::cerr << "socket failed" << std::endl;
        WSACleanup();
        exit(EXIT_FAILURE);
    }
    sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_);
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (::connect(client_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "connect failed" << std::endl;
        closesocket(client_fd_);
        WSACleanup();
        exit(EXIT_FAILURE);
    }
}

void SocketClient::send(const std::string& data) {
    ::send(client_fd_, data.c_str(), data.size(), 0);
}

std::string SocketClient::receive() {
	// 数据有可能一次接收不完，所以需要循环接收
	char buffer[8193] = { 0 };
	std::string received_data;
	int bytes_read;
	while ((bytes_read = recv(client_fd_, buffer, sizeof(buffer) - 1, 0)) > 0) {
		buffer[bytes_read] = '\0'; // 确保字符串结束
		received_data += buffer;
	}
	if (bytes_read < 0) {
		perror("recv failed");
		return "";
	}
	return received_data;
}
