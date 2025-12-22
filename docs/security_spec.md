# セキュリティ仕様（SSRF / 外部HTTP / RSS信頼境界）

**対象プロジェクト**: LatestInformationDiscussionAgents  
**目的**: ユーザー入力（URL/トピック）およびRSS許可リスト由来の外部HTTPアクセスに対し、SSRF・情報漏洩・DoSのリスクを最小化する。  
**適用範囲**: `src/agents/researcher.py`、`src/utils/rss.py`、（将来）`src/utils/security.py`、`src/ui/streamlit_app.py`、ログ/設定の運用。  

---

## 1. 脅威モデル（要約）

- **SSRF**: アプリがユーザー指定URLやRSS由来URLにアクセスできるため、`localhost` やVPC内、クラウドメタデータ（例: `169.254.169.254`）へ到達して情報を取得する可能性。
- **DNSリバインディング**: 一見外部に見えるホスト名が、解決時に内部IPへ向く。
- **オープンリダイレクト経由の到達**: 初回URLは安全でも、リダイレクトで内部へ到達。
- **DoS/リソース枯渇**: 巨大レスポンス、遅延レスポンス、巨大RSS/XML、無制限リダイレクトによるCPU/メモリ/帯域の消費。
- **ログによる情報漏洩**: URLクエリ等に含まれるトークン・識別子のログ保存。

---

## 2. セキュリティ基本方針（必須）

1. **外部HTTPアクセスの入口を1か所に集約**し、同一の検証・制限を必ず通す（URL直入力、RSSフィード取得、RSS item.link 取得のすべて）。
2. **許可スキーム・拒否IPレンジ・リダイレクト制御・サイズ制限**を標準装備する。
3. **デフォルトは最小権限（安全寄り）**とし、必要な場合のみ設定で緩和する。
4. **設定（環境変数）は信頼境界**。本番では「環境変数によるURL供給」を原則無効または厳格に検証する。

---

## 3. 設定仕様（環境変数）

### 3.1 URL直入力（topicがURL）の許可
- **`ALLOW_URL_FETCH`**: `0` / `1`（既定: `1`）
  - `0`: UIでURL入力を受けても、URL取得は拒否しRSS検索のみを許可（またはエラーを返す）
  - `1`: URL取得を許可（ただし本仕様の検証を通過した場合のみ）

### 3.2 スキーム許可
- **`URL_ALLOWED_SCHEMES`**: 例 `https`（既定: `https`）
  - 推奨: 本番は `https` のみ
  - 例外的に `http` を許可する場合でも、内部IP拒否は必須

### 3.3 リダイレクト
- **`URL_ALLOW_REDIRECTS`**: `0` / `1`（既定: `0`）
- **`URL_MAX_REDIRECTS`**: 整数（既定: `2`、ただし `URL_ALLOW_REDIRECTS=1` のときのみ有効）
  - `URL_ALLOW_REDIRECTS=1` の場合、**各ステップで宛先URL検証を再実行**する

### 3.4 ドメイン許可リスト（推奨）
- **`URL_ALLOWLIST_DOMAINS`**: カンマ区切り/空白区切りで複数（既定: 空＝無効）
  - 例: `example.com, www.nikkei.com, www3.nhk.or.jp`
  - 仕様:
    - **完全一致 or サブドメイン許可**（どちらにするかを明記）
    - 推奨: `*.example.com` のようなワイルドカードは **明示的に**許可する場合のみ

### 3.5 内部IP拒否（必須）
- **`URL_BLOCK_PRIVATE_IPS`**: `0` / `1`（既定: `1`）
  - `1`: ループバック/リンクローカル/プライベート/予約済み/マルチキャスト等を拒否

### 3.6 タイムアウト・サイズ制限
- **`HTTP_CONNECT_TIMEOUT_SEC`**: 既定 `3`
- **`HTTP_READ_TIMEOUT_SEC`**: 既定 `7`
- **`HTTP_MAX_BYTES`**: 既定 `5_000_000`（5MB）
  - RSSは `RSS_MAX_BYTES`（既定 2MB）など分離してもよい

### 3.7 RSS許可リストの優先順位（信頼境界）
- **`RSS_FEEDS_FILE_ONLY`**: `0` / `1`（既定: `1` 推奨）
  - `1`: `RSS_FEED_URLS` を無視し、`config/rss_feeds.txt` のみを利用
  - `0`: 現行通り `RSS_FEED_URLS` 優先を許可（ただし **URL検証必須**）

---

## 4. URL検証仕様（共通）

以下の検証を `validate_outbound_url()` として共通化し、外部HTTPアクセス前に必ず実行する。

### 4.1 パース・正規化
- `urlparse(url)` を実施し、以下を満たさない場合は拒否
  - `scheme` が存在し、`URL_ALLOWED_SCHEMES` 内
  - `hostname` が存在
- `userinfo`（`username`/`password`）が含まれる場合は拒否
- `port` は明示される場合のみ許可。必要なら許可ポートのホワイトリストを持つ（例: `443`、（HTTP許可なら）`80`）

### 4.2 ホスト名ベースの拒否（最低限）
- `localhost` / `localhost.` は拒否
- 可能なら `.local` / `.internal` 等のローカルTLD相当を拒否（運用要件による）

### 4.3 DNS解決 + IPレンジ拒否（必須）
ホスト名を解決し、返ってきた全てのIPに対して以下を適用する。

#### IPv4（拒否）
- `0.0.0.0/8`
- `10.0.0.0/8`
- `100.64.0.0/10`（CGNAT）
- `127.0.0.0/8`（loopback）
- `169.254.0.0/16`（link-local）
- `172.16.0.0/12`
- `192.0.0.0/24`
- `192.0.2.0/24`（TEST-NET-1）
- `192.168.0.0/16`
- `198.18.0.0/15`（benchmark）
- `198.51.100.0/24`（TEST-NET-2）
- `203.0.113.0/24`（TEST-NET-3）
- `224.0.0.0/4`（multicast）
- `240.0.0.0/4`（reserved）

#### IPv6（拒否）
- `::/128`（unspecified）
- `::1/128`（loopback）
- `::ffff:0:0/96`（IPv4-mapped 全体は要注意。実IPに展開して判定すること）
- `fe80::/10`（link-local）
- `fc00::/7`（unique local）
- `ff00::/8`（multicast）

> 注意: ここは「厳しめ」設定。要件により緩める場合でも、少なくとも loopback/ULA/link-local は拒否を維持すること。

### 4.4 リダイレクト
- `URL_ALLOW_REDIRECTS=0` の場合:
  - `requests.get(..., allow_redirects=False)` を使用
  - 30xが返ったらエラー（URL取得失敗）として扱う
- `URL_ALLOW_REDIRECTS=1` の場合:
  - 最大 `URL_MAX_REDIRECTS` 回まで
  - `Location` を解決して、**毎回 `validate_outbound_url()` を再実行**
  - 相対URLは結合してから検証

### 4.5 レスポンス制限（DoS対策）
- `timeout=(HTTP_CONNECT_TIMEOUT_SEC, HTTP_READ_TIMEOUT_SEC)` を適用
- `stream=True` で受信し、累計が `HTTP_MAX_BYTES` を超えたら中断
- 可能なら `Content-Length` が `HTTP_MAX_BYTES` 超なら即拒否
- HTML以外の `Content-Type` は拒否（または警告＋最小限の扱い）

---

## 5. RSS仕様（フィードと記事リンクの扱い）

### 5.1 RSSフィードURL（`rss_feeds.txt` / `RSS_FEED_URLS`）
- フィードURLは **URL検証（4章）を必須**とする
- `RSS_FEEDS_FILE_ONLY=1` を推奨し、運用で環境変数注入を避ける
- RSS取得は `RSS_MAX_BYTES`（例: 2MB）を別途設けることを推奨

### 5.2 RSS item.link（記事URL）
次のどちらかを明確に採用する。

- **A案（安全最優先・推奨）**:
  - 記事URLのドメインが「フィードURLのドメインと同一」または `URL_ALLOWLIST_DOMAINS` に含まれる場合のみ取得
  - それ以外はスキップ
- **B案（柔軟）**:
  - 記事URLは取得可。ただし **URL検証（4章）を必ず実行**し、内部宛先/危険宛先を拒否

#### 実装での切替（運用スイッチ）
現行実装では、環境変数で方針を切り替える。

- **`RSS_ITEM_LINK_POLICY`**: `A` / `B`（既定: `A`）
  - `A`: フィードと同一ドメイン（サブドメイン含む）または `URL_ALLOWLIST_DOMAINS` に含まれる場合のみ取得
  - `B`: 記事URLは取得可（ただしURL検証は必須）

---

## 6. ログ仕様（情報漏洩/ログ注入対策）

- **URLをログに残す場合**:
  - クエリ文字列（`?a=b`）やフラグメント（`#...`）は原則 **マスク/除去**
  - 改行や制御文字は除去
  - 長さは上限（例: 200文字）でトリム
- 例:
  - `https://example.com/path?token=...` → `https://example.com/path?…`
- LLMの生出力は必要最小限の断片のみ（すでに実装している「短いsnippet」方針を維持）

---

## 7. 例外（エラー）仕様

実装では専用例外を導入し、UIとログで扱いを分ける。

- `UrlValidationError`: URLが検証に失敗（危険/不正）
- `OutboundHttpError`: HTTPエラー（接続・タイムアウト・HTTPステータス等）
- `ResponseTooLargeError`: サイズ上限超過

UI表示は「危険URLは拒否しました」「RSSフィードが不正/危険です」など、**原因と対処**が分かる文言にする。

---

## 8. 実装配置（推奨）

- `src/utils/security.py`（新規）
  - `validate_outbound_url(url: str, *, purpose: Literal["article","rss"], ...) -> str`
  - `resolve_host_ips(hostname: str) -> list[str]`
  - `is_blocked_ip(ip: str) -> bool`
  - `sanitize_url_for_logging(url: str) -> str`
- `src/agents/researcher.py`
  - URL直入力時、RSS item.link 取得時に `validate_outbound_url()` を必ず通す
- `src/utils/rss.py`
  - フィードURLに `validate_outbound_url()` を必ず通す

---

## 9. テスト仕様（必須ケース）

### URL検証ユニットテスト
- 許可: `https://example.com/news`
- 拒否:
  - `http://127.0.0.1/`
  - `http://169.254.169.254/latest/meta-data/`
  - `https://localhost:11434/`
  - `file:///etc/passwd`
  - `https://user:pass@example.com/`
  - `https://[::1]/`
  - `https://example.com` → 302 → `http://127.0.0.1/`（リダイレクト検証）

### RSS関連
- `rss_feeds.txt` から読み込んだURLが危険なら除外される/例外になる
- RSS item.link がフィード外ドメインのとき、A案なら除外、B案なら検証に通れば取得

### サイズ上限
- `Content-Length` が上限超なら即拒否
- ストリーム読みで上限超過したら中断

---

## 10. 移行手順（推奨）

1. まず `validate_outbound_url()` を導入し、URL直入力とRSSフィード取得に適用（SSRFの入口を塞ぐ）
2. 次に RSS item.link に対して A案/B案を決定し適用
3. サイズ制限とログマスキングを追加
4. テスト追加 → スモークで実URLを数件確認


