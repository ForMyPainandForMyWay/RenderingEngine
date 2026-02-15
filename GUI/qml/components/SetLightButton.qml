import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic

Button {
    property string stext: "..."
    text: stext
    background: Rectangle {
        radius: tablePage.radius
        color: "lightgray"
    }
    contentItem: Text {
        text: parent.text
        color: "black"
        font.pixelSize: 12
        verticalAlignment: Text.AlignVCenter
    }
}