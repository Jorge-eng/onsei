

#include <zmq.h>
#include <map>
#include <iostream>
#include "log.h"

void *zmq_ctx_new ();
int zmq_ctx_destroy (void *context);
int zmq_recv (void *socket, void *buf, size_t len, int flags);

//example url "tcp://localhost:5563"

template <class Deserializer,class Message,class Subscriber>
class ZmqSubscriber {
public:
    ZmqSubscriber(size_t buf_size, const std::string & url)
    :_max_size(buf_size) {
        _message_buf = (uint8_t *)malloc(buf_size);
        _url = url;
    }
    
    ~ZmqSubscriber() {
        free(_message_buf);
        
        for (auto it = _subscribers.begin(); it != _subscribers.end(); it++) {
            delete (*it).second;
        }
    }
    
    void add_subscriber(const std::string & id, Subscriber * subscriber) {
        _subscribers[id] = subscriber;
    }
    
    void run() {

        void *context = zmq_ctx_new ();
        void *subscriber = zmq_socket (context, ZMQ_SUB);
        zmq_connect (subscriber, _url.c_str());
        zmq_setsockopt (subscriber, ZMQ_SUBSCRIBE,NULL, 0);

        while (1) {
            
            int size = zmq_recv (subscriber, (zmq_msg_t *)_message_buf, _max_size, 0);
            
            if (size == -1) {
                LOG("error receiving data");
            }
            
            _message_buf[size] = '\0';

            Message message;
            if (!_deserializer.deserialize_protobuf(_message_buf,size,message)) {
                std::cerr << "error parsing message" << std::endl;
                continue;
            }
            
            for (auto it = _subscribers.begin(); it != _subscribers.end(); it++) {
                (*it).second->receive_message(message);
            }
            
        }
        
        //  We never get here, but clean up anyhow
        zmq_close (subscriber);
        zmq_ctx_destroy (context);

    }
    
private:
    std::map<std::string,Subscriber *> _subscribers;
    Deserializer _deserializer;
    uint8_t * _message_buf;
    std::string _url;
    const size_t _max_size;
    
};
