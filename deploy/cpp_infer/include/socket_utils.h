#pragma once

#include <string>

class SocketServer {
public:
  SocketServer(int port);
  ~SocketServer();

  void start();
  std::string receive();
  void send(const std::string &data);

private:
  int port_;
  int server_fd_;
  int client_fd_;
  //struct sockaddr_in server_addr_;
  //struct sockaddr_in client_addr_;
  //socklen_t client_addr_len_;
};
