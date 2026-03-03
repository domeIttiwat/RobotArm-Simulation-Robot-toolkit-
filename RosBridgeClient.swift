// RosBridgeClient.swift
// Robot Toolkit — rosbridge v2 WebSocket client
//
// Manages a persistent WebSocket connection to a rosbridge v2 server
// (ws://<host>:9090) and exposes a simple publish API for ROS topics.
//
// Threading model
// ───────────────
// The class is @MainActor-isolated so all @Published mutations and
// public methods run on the main thread. URLSession completion handlers
// are bridged back with `Task { @MainActor in … }`.
//
// Wire protocol (rosbridge v2)
// ────────────────────────────
//  advertise  →  {"op":"advertise","topic":"/foo","type":"pkg/Type"}
//  publish    →  {"op":"publish","topic":"/foo","msg":{…}}
//  (incoming messages are received but ignored in this implementation)

import Foundation

// MARK: - ConnectionState

/// Lifecycle states for the rosbridge WebSocket.
enum ConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case error(String)
}

// MARK: - RosBridgeClient

/// Observable WebSocket client for rosbridge v2.
///
/// Usage:
/// ```swift
/// let ros = RosBridgeClient()
/// ros.connect(to: URL(string: "ws://192.168.1.100:9090")!)
/// ros.publish(topic: "/safety_status", type: "std_msgs/Int8", data: ["data": 2])
/// ros.disconnect()
/// ```
@MainActor
final class RosBridgeClient: ObservableObject {

    // MARK: Published

    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var isConnected = false

    // MARK: Private state

    private let session = URLSession(configuration: .default)
    private var socket: URLSessionWebSocketTask?
    private var reconnectTask: Task<Void, Never>?
    private var currentURL: URL?
    private var autoReconnect = false
    /// Topics already advertised in the current connection.
    private var advertisedTopics = Set<String>()

    private static let reconnectDelay: Duration = .seconds(5)

    // MARK: - Public API

    /// Opens a WebSocket connection and enables automatic reconnection on failure.
    ///
    /// - Parameter url: Full rosbridge WebSocket URL, e.g. `ws://192.168.1.100:9090`.
    func connect(to url: URL) {
        currentURL  = url
        autoReconnect = true
        cancelReconnect()
        openSocket()
    }

    /// Closes the connection gracefully and disables automatic reconnection.
    func disconnect() {
        autoReconnect = false
        cancelReconnect()
        closeSocket(with: .goingAway)
        applyState(.disconnected)
    }

    /// Publishes a message to a ROS topic via rosbridge v2.
    ///
    /// The topic is advertised automatically the first time it is used per
    /// connection. If the socket is not connected the call is silently ignored.
    ///
    /// - Parameters:
    ///   - topic: Full ROS topic name, e.g. `"/safety_status"`.
    ///   - type:  ROS message type string, e.g. `"std_msgs/Int8"`.
    ///   - data:  Message payload as a key-value dictionary, e.g. `["data": 2]`.
    func publish(topic: String, type: String, data: [String: Any]) {
        guard isConnected else { return }

        if !advertisedTopics.contains(topic) {
            sendJSON(["op": "advertise", "topic": topic, "type": type])
            advertisedTopics.insert(topic)
        }
        sendJSON(["op": "publish", "topic": topic, "msg": data])
    }

    // MARK: - Socket lifecycle

    private func openSocket() {
        guard let url = currentURL else { return }

        // Tear down any stale socket without triggering a reconnect.
        closeSocket(with: .normalClosure)
        applyState(.connecting)
        advertisedTopics.removeAll()

        let task = session.webSocketTask(with: url)
        socket = task
        task.resume()

        // Kick off the receive loop so we can detect disconnection.
        startReceiveLoop(for: task)
        // Confirm the handshake completed with a ping.
        confirmHandshake(for: task)
    }

    private func closeSocket(with code: URLSessionWebSocketTask.CloseCode) {
        socket?.cancel(with: code, reason: nil)
        socket = nil
    }

    /// Sends a ping immediately after resume to verify the WebSocket
    /// handshake succeeded before marking the connection as .connected.
    private func confirmHandshake(for task: URLSessionWebSocketTask) {
        task.sendPing { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self, self.socket === task else { return }
                if let error {
                    self.handleError(error, from: task)
                } else {
                    self.applyState(.connected)
                }
            }
        }
    }

    /// Recursive receive loop — keeps the socket alive and detects remote closure.
    private func startReceiveLoop(for task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self, self.socket === task else { return }
                switch result {
                case .success:
                    self.startReceiveLoop(for: task)   // keep listening
                case .failure(let error):
                    self.handleError(error, from: task)
                }
            }
        }
    }

    // MARK: - State helpers

    private func applyState(_ newState: ConnectionState) {
        connectionState = newState
        isConnected = (newState == .connected)
    }

    private func handleError(_ error: Error, from task: URLSessionWebSocketTask) {
        guard socket === task else { return }   // ignore stale callbacks
        socket = nil
        advertisedTopics.removeAll()
        applyState(.error(error.localizedDescription))
        if autoReconnect { scheduleReconnect() }
    }

    // MARK: - Auto-reconnect

    private func scheduleReconnect() {
        cancelReconnect()
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(for: Self.reconnectDelay)
            guard let self else { return }
            await MainActor.run {
                guard self.autoReconnect else { return }
                self.openSocket()
            }
        }
    }

    private func cancelReconnect() {
        reconnectTask?.cancel()
        reconnectTask = nil
    }

    // MARK: - JSON send

    private func sendJSON(_ dict: [String: Any]) {
        guard let task  = socket,
              let data  = try? JSONSerialization.data(withJSONObject: dict),
              let text  = String(data: data, encoding: .utf8) else { return }

        task.send(.string(text)) { [weak self] error in
            guard let error else { return }
            Task { @MainActor [weak self] in
                guard let self, self.socket === task else { return }
                self.handleError(error, from: task)
            }
        }
    }
}
