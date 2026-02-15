import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    width: parent.width
    height: parent.height
    spacing: 20
    property var hRatio: 0.1

    Rectangle {
        Layout.preferredWidth: parent.width
        Layout.preferredHeight: parent.height * parent.hRatio
        color: "transparent"
        Slider{
            id: fovSlide
            value: 0.5
            width: parent.width * 0.8
            height: parent.height * 0.2
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            onValueChanged: {
                settingProxy.setFOV(fovSlide.value.toFixed(2));
            }
        }
        Text{
            text: "FOV = " + fovSlide.value.toFixed(2)
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: fovSlide.bottom
            anchors.margins: 8
        }
    }

    Rectangle {
        Layout.preferredWidth: parent.width
        Layout.preferredHeight: parent.height * parent.hRatio
        color: "transparent"
        Slider{
            id: nearPlaneSlide
            value: 0.5
            width: parent.width * 0.8
            height: parent.height * 0.2
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            onValueChanged: {
                settingProxy.setNear(nearPlaneSlide.value.toFixed(2));
            }
        }
        Text{
            text: "Near Plane = " + nearPlaneSlide.value.toFixed(2)
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: nearPlaneSlide.bottom
            anchors.margins: 8
        }
    }

    Rectangle {
        Layout.preferredWidth: parent.width
        Layout.preferredHeight: parent.height * parent.hRatio
        color: "transparent"
        Slider{
            id: farPlaneSlide
            value: 0.5
            width: parent.width * 0.8
            height: parent.height * 0.2
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            onValueChanged: {
                settingProxy.setFar(farPlaneSlide.value.toFixed(2));
            }
        }
        Text{
            text: "Far Plane = " + farPlaneSlide.value.toFixed(2)
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: farPlaneSlide.bottom
            anchors.margins: 8
        }
    }
    Item {
        Layout.fillHeight: true  // 占据所有剩余垂直空间
    }
}