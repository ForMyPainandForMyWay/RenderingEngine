import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.platform

// 导入基础组件
import "../components"

Column {
    anchors.fill: parent
    // 颜色对话框
    ColorDialog {
        id: colorDialog
        title: "Select Light Color"
        currentColor: Qt.rgba(1,1,1,1)
        property var applyCallback

        onAccepted: {
            if (applyCallback)
                applyCallback(color)
        }
    }

    RowLayout {
        width: parent.width
        CheckLight {
            stext: "Spot Light"
            Layout.alignment: Qt.AlignLeft
            onCheckedChanged: {
                settingProxy.enableSpot(checked)
            }
        }
        SetLightButton {
            id: spotColorSelect
            Layout.preferredHeight: 15
            Layout.alignment: Qt.AlignRight
            property color keyColor: Qt.white
            onClicked: {
                colorDialog.color = keyColor
                colorDialog.applyCallback = c => keyColor = c
                colorDialog.open()
            }
        }
    }
    RowLayout {
        width: parent.width
        CheckLight {
            stext: "Env Light"
            Layout.alignment: Qt.AlignLeft
            onCheckedChanged: {
                settingProxy.enableEnv(checked)
            }
        }
        SetLightButton {
            id: envColorSelect
            Layout.preferredHeight: 15
            Layout.alignment: Qt.AlignRight
            property color keyColor: Qt.white
            onClicked: {
                colorDialog.color = keyColor
                colorDialog.applyCallback = c => keyColor = c
                colorDialog.open()
            }
        }
    }
    RowLayout {
        width: parent.width
        CheckLight {
            stext: "Point Light"
            Layout.alignment: Qt.AlignLeft
            onCheckedChanged: {
                settingProxy.enablePoint(checked)
            }
        }
        SetLightButton {
            id: pointColorSelect
            Layout.preferredHeight: 15
            Layout.alignment: Qt.AlignRight
            property color keyColor: Qt.white
            onClicked: {
                colorDialog.color = keyColor
                colorDialog.applyCallback = c => keyColor = c
                colorDialog.open()
            }        }
    }
}