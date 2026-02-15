import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    width: parent.width
    height: parent.height
    spacing: 0
    property var hRatio: 0.2

    Rectangle {
        Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom
        Layout.preferredWidth: parent.width
        Layout.preferredHeight: parent.height * parent.hRatio
        color: "transparent"
        Slider{
            id: pitchSlide
            value: 0.5
            width: parent.width * 0.8
            height: parent.height * 0.2
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            onValueChanged: {
                settingProxy.setPitch(pitchSlide.value.toFixed(2))  // 注意这里保留了两位小数
            }
        }
        Text{
            text: "Pitch = " + pitchSlide.value.toFixed(2)
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: pitchSlide.bottom
            anchors.margins: 8
        }
    }

    Rectangle {
        Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom
        Layout.preferredWidth: parent.width
        Layout.preferredHeight: parent.height * (1 - parent.hRatio)
        color: "transparent"
        Dial{
            id: yawDial
            value: 0.5
            height: parent.width
            width: parent.width
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            onValueChanged: {
                settingProxy.setYaw(yawDial.value.toFixed(2))  // 注意两位小数
            }
        }
        Text{
            text: "Yaw = " + yawDial.value.toFixed(2)
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: yawDial.bottom
            anchors.margins: -5
        }
    }
}