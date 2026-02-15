import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic

CheckBox {
    id: root

    property int boxSize: 20
    property int textSize: 14
    property string stext: "Option"

    spacing: 8

    /* 屏蔽indicator */
    indicator: null

    /* 所有可视内容都放进 contentItem */
    contentItem: Row {
        id: row
        spacing: root.spacing
        anchors.verticalCenter: parent.verticalCenter

        Rectangle {
            width: root.boxSize
            height: root.boxSize
            radius: 4
            border.width: 2
            border.color: "black"
            color: root.checked ? "black" : "#dddddd"

            Text {
                anchors.centerIn: parent
                text: "✓"
                color: "white"
                font.pixelSize: parent.height * 0.7
                font.bold: true
                visible: root.checked
            }
        }

        Text {
            text: root.stext
            font.pixelSize: root.textSize
            color: root.checked ? "black" : "gray"
            verticalAlignment: Text.AlignVCenter
        }
    }
}