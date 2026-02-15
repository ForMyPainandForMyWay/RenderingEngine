import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Column {
    id: root
    anchors.fill: parent
    RowLayout {
        width: parent.width
        Text{
            text: "Triangle Nums: "
            Layout.alignment: Qt.AlignLeft
        }
        Text{
            text: settingProxy.TriangleNums
            Layout.alignment: Qt.AlignRight
        }
    }

    RowLayout {
        width: parent.width
        Text{
            text: "Vertex Nums: "
            Layout.alignment: Qt.AlignLeft
        }
        Text{
            text: settingProxy.VertexNums
            Layout.alignment: Qt.AlignRight
        }
    }
}