// SafetyPanel.swift
// Robot Toolkit — safety control panel (rosbridge v2)
//
// Sends Int8 safety commands to the ROS topic /safety_status:
//   0 = Normal   1 = Warning   2 = Emergency Stop
//
// Designed for iPad — all buttons are at minimum 80 pt tall with
// large icons and Thai/English bilingual labels.
//
// Requires: RosBridgeClient.swift
// Targets:  iOS 16+  •  macOS 13+

import SwiftUI

// MARK: - SafetyPanel

/// Full-screen safety control panel.
///
/// Place inside a `NavigationStack` for the large title:
/// ```swift
/// NavigationStack { SafetyPanel() }
/// ```
struct SafetyPanel: View {

    @StateObject private var ros = RosBridgeClient()

    // MARK: State

    /// Robot IP address entered by the user.
    @State private var ipAddress = "192.168.1.100"
    private let port = 9090

    /// Last safety command sent (-1 = nothing sent yet).
    @State private var currentStatus: Int8 = -1

    /// Non-nil while a toast notification is visible.
    @State private var toastMessage: String? = nil

    // MARK: Body

    var body: some View {
        ZStack(alignment: .bottom) {
            Color(.systemGroupedBackground).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 24) {
                    connectionCard
                    Divider().padding(.horizontal)
                    safetyButtonStack
                    if currentStatus >= 0 {
                        statusFooter
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }
                    Spacer(minLength: 20)
                }
                .padding()
                .animation(.easeInOut(duration: 0.3), value: currentStatus)
            }

            // ── Toast overlay ──
            if let msg = toastMessage {
                toastView(msg)
                    .padding(.bottom, 40)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: toastMessage)
        .navigationTitle("Safety Panel")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.large)
        #endif
    }

    // MARK: - Connection card

    private var connectionCard: some View {
        VStack(spacing: 14) {

            // ── Status row ──
            HStack(spacing: 8) {
                // Animated status dot
                Circle()
                    .fill(dotColor)
                    .frame(width: 12, height: 12)
                    .shadow(color: dotColor.opacity(0.55), radius: 4)
                    .animation(.easeInOut(duration: 0.3), value: ros.connectionState)

                Text(stateLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Spacer()
            }

            // ── IP field + Connect button ──
            HStack(spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "network")
                        .foregroundStyle(.secondary)

                    TextField("Robot IP", text: $ipAddress)
                        .autocorrectionDisabled()
                        .submitLabel(.done)
                        #if os(iOS)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        #endif
                }
                .padding(10)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))

                Text(":\(port)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)

                Spacer()

                Button(action: toggleConnection) {
                    Label(
                        ros.isConnected ? "Disconnect" : "Connect",
                        systemImage: ros.isConnected ? "wifi.slash" : "wifi"
                    )
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 18)
                    .padding(.vertical, 10)
                    .background(
                        ros.isConnected
                            ? Color.red.opacity(0.12)
                            : Color.accentColor.opacity(0.12),
                        in: Capsule()
                    )
                    .foregroundStyle(ros.isConnected ? .red : .accentColor)
                    .overlay(
                        Capsule().stroke(
                            ros.isConnected ? Color.red : Color.accentColor,
                            lineWidth: 1
                        )
                    )
                }
                .disabled(isConnecting)
            }
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Safety buttons

    private var safetyButtonStack: some View {
        VStack(spacing: 14) {

            // 0 — Normal / Reset
            SafetyButton(
                label:     "NORMAL",
                subtitle:  "กลับสู่สถานะปกติ",
                icon:      "checkmark.shield.fill",
                color:     .safetyGreen,
                value:     0,
                current:   currentStatus,
                isEnabled: ros.isConnected
            ) {
                sendCommand(value: 0, label: "NORMAL")
            }

            // 1 — Warning
            SafetyButton(
                label:     "WARNING",
                subtitle:  "ลดความเร็ว / ระวัง",
                icon:      "exclamationmark.triangle.fill",
                color:     .safetyOrange,
                value:     1,
                current:   currentStatus,
                isEnabled: ros.isConnected
            ) {
                sendCommand(value: 1, label: "WARNING")
            }

            // 2 — Emergency Stop  (larger font + pulse animation when active)
            SafetyButton(
                label:     "EMERGENCY STOP",
                subtitle:  "หยุดฉุกเฉิน",
                icon:      "xmark.octagon.fill",
                color:     .safetyRed,
                value:     2,
                current:   currentStatus,
                isEnabled: ros.isConnected,
                isLarge:   true
            ) {
                sendCommand(value: 2, label: "EMERGENCY STOP")
            }
        }
    }

    // MARK: - Status footer

    private var statusFooter: some View {
        HStack(spacing: 12) {
            Image(systemName: statusIcon)
                .font(.title2)

            VStack(alignment: .leading, spacing: 2) {
                Text("สถานะปัจจุบัน")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(statusText)
                    .font(.headline.weight(.bold))
            }
            Spacer()
        }
        .foregroundStyle(statusColor)
        .padding(16)
        .background(statusColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(statusColor.opacity(0.35), lineWidth: 1)
        )
    }

    // MARK: - Toast

    private func toastView(_ message: String) -> some View {
        Text(message)
            .font(.subheadline.weight(.medium))
            .padding(.horizontal, 24)
            .padding(.vertical, 13)
            .background(.ultraThinMaterial, in: Capsule())
            .shadow(color: .black.opacity(0.15), radius: 12, y: 4)
    }

    // MARK: - Actions

    private func toggleConnection() {
        if ros.isConnected {
            ros.disconnect()
        } else {
            let host = ipAddress.trimmingCharacters(in: .whitespaces)
            guard !host.isEmpty,
                  let url = URL(string: "ws://\(host):\(port)") else { return }
            ros.connect(to: url)
        }
    }

    private func sendCommand(value: Int8, label: String) {
        ros.publish(
            topic: "/safety_status",
            type:  "std_msgs/Int8",
            data:  ["data": value]
        )
        currentStatus = value
        showToast("ส่งคำสั่ง: \(label)")
    }

    private func showToast(_ message: String) {
        withAnimation { toastMessage = message }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            withAnimation { toastMessage = nil }
        }
    }

    // MARK: - Computed helpers

    private var isConnecting: Bool {
        if case .connecting = ros.connectionState { return true }
        return false
    }

    private var dotColor: Color {
        switch ros.connectionState {
        case .connected:    return .green
        case .connecting:   return .yellow
        case .error:        return .red
        case .disconnected: return .gray
        }
    }

    private var stateLabel: String {
        switch ros.connectionState {
        case .connected:      return "Connected  •  ws://\(ipAddress):\(port)"
        case .connecting:     return "Connecting to \(ipAddress)…"
        case .error(let msg): return "Error: \(msg)"
        case .disconnected:   return "Not connected"
        }
    }

    private var statusIcon: String {
        switch currentStatus {
        case 0:  return "checkmark.shield.fill"
        case 1:  return "exclamationmark.triangle.fill"
        case 2:  return "xmark.octagon.fill"
        default: return "questionmark.circle"
        }
    }

    private var statusText: String {
        switch currentStatus {
        case 0:  return "ปกติ"
        case 1:  return "เตือน"
        case 2:  return "ฉุกเฉิน!"
        default: return "–"
        }
    }

    private var statusColor: Color {
        switch currentStatus {
        case 0:  return .safetyGreen
        case 1:  return .safetyOrange
        case 2:  return .safetyRed
        default: return .gray
        }
    }
}

// MARK: - SafetyButton

/// A single large safety action button.
///
/// - Highlights with a filled background when `value == current`.
/// - Pulses (scale + shadow glow) for the Emergency Stop (value == 2) when active.
/// - Grays out completely when `isEnabled` is false.
private struct SafetyButton: View {

    let label:     String
    let subtitle:  String
    let icon:      String
    let color:     Color
    /// The Int8 value this button publishes.
    let value:     Int8
    /// The most recently sent value (drives active state).
    let current:   Int8
    let isEnabled: Bool
    var isLarge:   Bool = false
    let action:    () -> Void

    // Pulse animation driven by onChange
    @State private var pulsing = false

    private var isActive:      Bool { current == value }
    /// Only the Emergency Stop button pulses when active.
    private var shouldPulse:   Bool { isActive && value == 2 }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 18) {

                // ── Icon ──
                Image(systemName: icon)
                    .font(.system(size: isLarge ? 46 : 36, weight: .bold))
                    .foregroundStyle(contentColor)
                    .frame(width: isLarge ? 58 : 46)

                // ── Labels ──
                VStack(alignment: .leading, spacing: 4) {
                    Text(label)
                        .font(isLarge
                              ? .system(size: 26, weight: .heavy, design: .rounded)
                              : .system(size: 20, weight: .bold,  design: .rounded))
                        .foregroundStyle(contentColor)

                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(subtitleColor)
                }

                Spacer()

                // ── Active indicator ──
                if isActive {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.white)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .padding(.horizontal, 22)
            .padding(.vertical, isLarge ? 22 : 18)
            .frame(maxWidth: .infinity, minHeight: 80)
            // Background fill
            .background(
                RoundedRectangle(cornerRadius: 18)
                    .fill(backgroundColor)
            )
            // Border (visible only when inactive)
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(
                        isEnabled ? color : Color.gray.opacity(0.35),
                        lineWidth: isActive ? 0 : 2
                    )
            )
            // Pulse: shadow glow
            .shadow(
                color: isEnabled && isActive ? color.opacity(pulsing ? 0.60 : 0.18) : .clear,
                radius: pulsing ? 18 : 6
            )
            // Pulse: scale
            .scaleEffect(pulsing ? 1.025 : 1.0)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .animation(.easeInOut(duration: 0.2), value: isActive)
        // Start / stop the pulse when shouldPulse changes
        .onChange(of: shouldPulse) { newValue in
            if newValue {
                withAnimation(.easeInOut(duration: 0.85).repeatForever(autoreverses: true)) {
                    pulsing = true
                }
            } else {
                withAnimation(.easeOut(duration: 0.3)) {
                    pulsing = false
                }
            }
        }
        .onAppear {
            if shouldPulse {
                withAnimation(.easeInOut(duration: 0.85).repeatForever(autoreverses: true)) {
                    pulsing = true
                }
            }
        }
    }

    // MARK: Derived colors

    private var backgroundColor: Color {
        guard isEnabled else { return Color.gray.opacity(0.08) }
        return isActive ? color : color.opacity(0.08)
    }

    private var contentColor: Color {
        guard isEnabled else { return Color.gray.opacity(0.45) }
        return isActive ? .white : color
    }

    private var subtitleColor: Color {
        guard isEnabled else { return Color.gray.opacity(0.35) }
        return isActive ? .white.opacity(0.82) : color.opacity(0.72)
    }
}

// MARK: - Color palette

private extension Color {
    /// iOS green  #34C759
    static let safetyGreen  = Color(red: 0.204, green: 0.780, blue: 0.349)
    /// iOS orange #FF9500
    static let safetyOrange = Color(red: 1.000, green: 0.584, blue: 0.000)
    /// iOS red    #FF3B30
    static let safetyRed    = Color(red: 1.000, green: 0.231, blue: 0.188)
}

// MARK: - Preview

#Preview("Safety Panel") {
    NavigationStack {
        SafetyPanel()
    }
}
