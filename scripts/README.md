# ZMQ Client-Server Scripts

This section contains server and client (robot) communication scripts utilizing ZeroMQ (ZMQ) for sending images and tracking latency. The server listens for incoming messages, while the client captures and sends images along with a timestamp to measure communication latency.

## Directories

### 1. `server/`
**Scripts:**
- `serverInd.py`
- `serverPickle.py`

**Description:**
These scripts handle the transmission of timestamp, color images, and depth images. serverInd.py does this as **raw binary data**. Each data component (timestamp, color image, depth image) is transmitted separately using ZMQ's socket communication. The server receives each piece of data and processes it independently. 
serverPickle.py packs the timestamp, color image, and depth image into a dictionary, then **serializes the entire dictionary using pickle**. The client sends the serialized message as a single entity, and the server deserializes it upon receipt. This method simplifies data handling but incurs a slight performance cost due to the serialization/deserialization process.

**Usage:**
Run the server to listen for incoming data, this will capture and benchmark the transmissions from the client.

---

### 2. `client/`
**Scripts:**
- `testClient.py`

**Description:**
These scripts are intended for running on the client (robot arm) machine, which will send a live feed of images to the server to be benchmarked.

**Usage:**
After the server is running, run the Client scripts to send Client information across the network.
