/**
 * スマホ用の画面（ウェブアプリ）。このスプレッドシートに紐づくスクリプトとして貼り付ける。
 * ここは薄く：タブ「app_data」を読んで文字のまま返すだけ。表示のロジックは index.html にある。
 * シート ID は書かない（紐づいているシートを getActiveSpreadsheet() で読む）。
 */
var APP_DATA_SHEET = 'app_data';

function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
    .setTitle('当日の買い目')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

/** タブ app_data の A 列（分けて書いた JSON）をつないで返す。無ければ空文字。 */
function getAppData() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(APP_DATA_SHEET);
  if (!sheet) return '';
  var rows = sheet.getLastRow();
  if (rows < 1) return '';
  return sheet.getRange(1, 1, rows, 1).getValues().map(function (r) { return String(r[0]); }).join('');
}
