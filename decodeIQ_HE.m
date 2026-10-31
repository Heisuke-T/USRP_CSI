%% decodeIQ_HE.m
% =========================================================================
%  [第2段] キャプチャ済み IQ をオフライン復号し、指定SSIDのCSIを記録する
%          (Wi-Fi 6 / 802.11ax = HE 対応版)
% -------------------------------------------------------------------------
%  概要:
%    captureIQ.m が HDD に保存した生 IQ を読み込み、WLAN Toolbox で
%    パケットを検出・復号して、
%      1) Beacon/Probe Response から SSID と BSSID の対応を学習
%      2) 復号できた全パケットについて、プリアンブル (Non-HTは L-LTF、
%         HTは HT-LTF、VHTは VHT-LTF、HEは HE-LTF) から
%         CSI (チャネル応答 H(k)) を算出
%      3) targetSSID に対応する BSSID のパケットのCSIだけを抽出
%    して .mat 保存する。
%
%    Wi-Fi 5 までの復号器 (旧 decodeIQ_VHT.m) と同じ構成・入出力形式・
%    SSID学習の方針に、HE (802.11ax / Wi-Fi 6) の復号経路を追加したもの。
%    Non-HT / HT / VHT もこのファイルで復号できる。
%
%  対応フォーマット:
%    Non-HT (802.11a/g レガシー) / HT-Mixed (802.11n / Wi-Fi 4) /
%    SU-VHT (802.11ac / Wi-Fi 5) / HE-SU・HE-EXT-SU (802.11ax / Wi-Fi 6)
%
%    Non-HT を残しているのは必須のため: 5GHz 帯の Beacon / Probe Response は
%    互換性のため常にレガシーレート (Non-HT) で送信されるので、Non-HT を
%    復号できないと SSID と BSSID の対応を学習できない。
%
%    SISO・20MHz 受信であることによる制約 (USRP B205 mini-i は受信アンテナ1本):
%      - 空間ストリーム数 = 1 のパケットのみ復号します。
%        マルチストリームは1本アンテナでは原理的に復号できません。
%      - 20MHz動作のパケットのみ対象です。40/80/160MHzで送信された
%        パケットは本機の受信帯域(20MHz)では正しく復号できません。
%      - STBC を使用しているパケットは非対応としてスキップします。
%
%    HE のうち非対応のフォーマット (検出・集計のみ):
%      - HE-MU  … OFDMA / MU-MIMO。復号には HE-SIG-B から RU 割り当てと
%                 ユーザ情報を読む必要があり、SISO 受信の前提から外れる。
%      - HE-TB  … トリガベース (上りOFDMA)。送信パラメータは AP が送った
%                 Trigger フレーム側にあり、HE-TB PPDU 単体では決まらない。
%      いずれも [HE 詳細] の「非対応の理由」に件数が出ます。
%
%  HE の復号手順 (MathWorks "Recovery Procedure for an 802.11ax Packet" 準拠):
%      L-STF → 粗CFO           (wlanCoarseCFOEstimate)
%      L-LTF → 精CFO・チャネル/雑音推定
%      L-SIG+RL-SIG+HE-SIG-A の4シンボル → フォーマット判定 (wlanFormatDetect)
%      L-SIG+RL-SIG → pre-HE チャネル推定 (wlanPreHEChannelEstimate) と
%                     L-SIG LENGTH の取得 (wlanLSIGBitRecover)
%      HE-SIG-A     → wlanHEDemodulate → wlanHEEqualize
%                     → wlanHESIGABitRecover → interpretHESIGABits
%                     (wlanHERecoveryConfig に MCS/GI/NSTS 等が入る)
%      HE-LTF       → wlanHELTFChannelEstimate  ← これが CSI
%      HE-Data      → wlanHEDemodulate → wlanHEEqualize
%                     → wlanHEDataBitRecover → A-MPDU分解 → MPDU復号
%
%    ※HE では VHT と違い SIG フィールドのビット解釈を自前実装する必要が
%      ありません (interpretHESIGABits が WLAN Toolbox にあるため)。
%      HT-SIG / VHT-SIG-A の自前パーサは Non-HT/HT/VHT 経路のために残して
%      あります。
%
%  既知の簡略化 (復号率が伸びないときに見直す箇所):
%    - 位相追跡は wlanHETrackPilotError による共通位相誤差 (CPE) の補正
%      までです。サブキャリアごとに傾く成分 (SFO によるタイミングドリフト)
%      までは追い切れないため、非常に長いパケットの末尾では誤差が残ります。
%    - 雑音分散は L-LTF から求めた 1 つの値をすべてのフィールドに使って
%      います。HE-Data はサブキャリア間隔が 1/4 (78.125kHz) なので厳密には
%      スケールが一致しません。「HE-SIG-A は解釈できるのに HE-Data の FCS が
%      通らない」場合はここを疑ってください。
%
%  必要環境:
%    - MATLAB / Communications Toolbox / WLAN Toolbox
%      HE 復号には R2019a 以降 (wlanHEDemodulate / wlanHERecoveryConfig 等)
%      が必要です。wlanPreHEChannelEstimate が無い版では、L-LTF の推定値を
%      pre-HE のサブキャリア本数へ広げる簡易版に自動でフォールバックします。
%    - captureIQ.m が出力した *_raw.mat
%
%  入力ファイル (どちらでもよい。指定が無ければ最新のものを自動選択):
%    <hddInputPath>/<日時>_raw.mat   captureIQ.m の出力 (complex double)
%    <hddInputPath>/<日時>_raw.bin   captureIQ_single.m の出力 (int16 の連続
%                                    ストリーム。取得条件は同名の
%                                    <日時>_rawmeta.mat から読む)
%
%    .bin は必要なところだけを 1 秒ずつ読み進めるので、キャプチャが
%    何分あってもメモリ使用量は一定 (約 320 MB) のままです。
%    フォルダ内のファイルをまとめて復号したいときは decodeIQ_HE_batch.m を
%    使う (このスクリプトを 1 ファイルずつ呼び出す)。
%
%    そのため captureIQ_single.m の writeSegments は false のままでよく、
%    *_segNN_raw.mat に分割する必要はありません。1 キャプチャ = 1 ファイルで
%    扱えます。
%
%  出力ファイル (HDD と USB メモリの両方に同じ内容を保存):
%    <hddSavePath>/<yyyymmddHHMM>_<SSID>_CSI.mat
%    <usbSavePath>/<yyyymmddHHMM>_<SSID>_CSI.mat
%      ※ファイル名の日時はキャプチャ時刻 (復号時刻ではない)。元の生IQと
%        対応が取れるようにするため。
%
%    [主データ] ResultCSI.m がそのまま読める形式
%      csi               … [パケット数 x サブキャリア数] complex の行列
%      subcarrierIndices … [1 x サブキャリア数] 使用サブキャリア番号 k
%      packetStartIndex  … [パケット数 x 1] 各パケットのサンプル位置
%      csiMeta           … 処理条件・復号統計 (sampleRate 等を含む)
%      ※ Non-HT/HT は 52 本、VHT(20MHz) は 56 本、HE(20MHz, 242-tone RU) は
%         242 本とサブキャリア数が異なるため 1 つの行列には混在させられない。
%         本スクリプトは HE 用なので、HE のパケットが 1 件でもあれば HE を
%         主データとし、無ければパケット数が最多の形式を採用する。
%         どれを採用したかは csiMeta.primaryFormat に記録する。
%
%    [フォーマット別] 複数形式が必要な場合はこちらを使う
%      csiNonHT / timeSecNonHT / frameTypeNonHT / fcsNonHT
%      csiHT    / timeSecHT    / frameTypeHT    / fcsHT
%      csiVHT   / timeSecVHT   / frameTypeVHT   / fcsVHT
%      csiHE    / timeSecHE    / frameTypeHE    / fcsHE
%      uplinkHE   … [パケット数 x 1] true=上り(STA->AP), false=下り(AP->STA)
%                   HE-SIG-A の UL/DL ビット由来。上りと下りは送信機が違う
%                   ので CSI も別物。解析時は必ず分けること。
%      bssColorHE … [パケット数 x 1] HE-SIG-A の BSS Color
%      subcarrierIndicesNonHT … [1 x 52]   Non-HT の使用サブキャリア番号
%      subcarrierIndicesHT20  … [1 x 52]   HT-20MHz の使用サブキャリア番号
%      subcarrierIndicesVHT20 … [1 x 56]   VHT-20MHz の使用サブキャリア番号
%      subcarrierIndicesHE20  … [1 x 242]  HE-20MHz の使用サブキャリア番号
%
%    [補助]
%      timeSec      … [パケット数 x 1] キャプチャ開始からの相対時刻 [s]
%      frameType    … [パケット数 x 1] 各パケットのフレーム種別
%      phyFormat    … [パケット数 x 1] 'Non-HT' / 'HT' / 'VHT' / 'HE'
%      fcsVerified  … [パケット数 x 1] false は FCS 未検証 (MAC ヘッダのみ
%                      から送信元を判定したもの)。CSI 自体には影響しない。
%      targetSSIDOut / targetBSSIDOut … 指定SSIDと学習されたBSSID
%      seenNetworks … 検出できた全 BSSID/SSID の一覧 (診断用)
%      frameLog     … 復号できた全フレーム (対象 SSID 以外も含む) の時刻・送信元・宛先。
%                     IMU 計測アプリ (WT9011DCL) がノート PC から送る Wi-Fi の合図
%                     (上りのデータフレーム) を後で探し、IMU と時刻を合わせるのに使う。
%                     数値だけの構造体なので Python (h5py) からも読みやすい:
%                       timeSec  … キャプチャ開始からの時刻 [s] (timeSec* と同じ時間軸)
%                       ta / ra  … 送信元 / 宛先の MAC アドレスを 48 bit の数値にしたもの (NaN = 不明)
%                       phy      … 1=Non-HT 2=HT 3=VHT 4=HE
%                       isData   … フレーム種別に "Data" を含むか
%                       toAP     … 宛先が対象 BSSID (= AP 宛ての上り)
%                       fcsVerified, uplinkHE … 上の同名の変数と同じ意味
%
%    ※ 対象 SSID への帰属判定は、送信元(Address2)・宛先(Address1)の
%      どちらかが対象 BSSID と一致するかで行う (ダウンリンクは Address2、
%      アップリンクは Address1 が BSSID になるため)。
%
%  処理時間の目安:
%    電波の混雑度に強く依存します。まずは captureDuration を短く (1s 程度)
%    して動作確認することを推奨します。
%
%    復号が遅いときの主な要因は次の3つです。
%      1) キャプチャ長そのもの
%         パケット検出は全サンプルを走査するので、キャプチャが長いほど
%         比例して時間がかかります。
%      2) HE のデータ復号 (LDPC)
%         HE は LDPC が既定で、MATLAB の LDPC 復号は反復計算のため
%         BCC (Viterbi) より大幅に遅くなります。さらに HE は A-MPDU で
%         数万バイトまとめて送るのでパケット 1 個が長くなります。
%      3) 誤検出パケットの復号試行
%         pktDetThreshold を下げすぎると雑音を拾って復号を試み続けます。
%         復号サマリの「パケット検出数」に対して「MAC まで復号成功」が
%         極端に少ない場合は、しきい値を上げる方が速くなります。
% =========================================================================

clear;
% 一括処理 (decodeIQ_HE_batch.m) から呼ばれているときは画面を消さない。
% 前のファイルの結果や全体の進捗が流れて見えなくなるため。
if isempty(getappdata(0, 'decodeIQ_HE_batch'))
    clc;
end

%% ------------------------------------------------------------------------
%  1. ユーザ設定パラメータ
%  ------------------------------------------------------------------------

% --- 入力元 (ホストPC に接続された HDD。captureIQ.m の保存先と揃える) ----
%     現在の環境: HDPC-UT (D:)
hddInputPath = 'D:\IQ_raw';

% --- 読み込む生IQファイル ------------------------------------------------
%     '' にすると hddInputPath 内で最も新しい *_raw.mat を自動選択する。
%     特定のファイルを指定したい場合はファイル名かフルパスを書く。
inputRawFile = '09272233_raw.mat';

% --- 出力先 --------------------------------------------------------------
%     同じ内容を HDD と USB メモリの両方へ保存する。
%     HDD は HDPC-UT (D:)。
hddSavePath = 'D:\IQ_csi';

%     USB メモリのドライブレターをここに設定する (例: 'F:\IQ')。
%     '' のままにすると USB への保存はスキップし、HDD にのみ保存する。
%     ※ResultCSI.m の既定の読み込み先は 'D:\IQ' なので、USB を使わない
%       場合は ResultCSI.m 側の usbSavePath を 'D:\IQ_csi' に変更すること。
usbSavePath = '';

% --- 抽出したい Wi-Fi の SSID --------------------------------------------
targetSSID = 'WAX202';

% --- 復号パラメータ ------------------------------------------------------
chanBW          = 'CBW20';      % WLAN Toolbox のチャネル帯域幅指定
                                 % (キャプチャ時の 20MHz に対応)
pktDetThreshold = 0.5;          % wlanPacketDetect のしきい値 (0〜1)
                                 % 下げると弱いパケットも拾うが誤検出が増え、
                                 % 復号時間も伸びる
verboseErrors   = false;        % true にすると復号エラーを毎回表示する
                                 % (通常は最後に集計のみ表示)
showEachPacket  = true;         % true: 復号できたパケットを 1 行ずつ表示する
                                 %       (数千行になる)。結果のサマリだけ
                                 %       見たい・残したいときは false
saveDecodeLog   = true;         % true: 画面に表示した内容を、出力 (*_CSI.mat)
                                 %       と同じフォルダに *_CSI_log.txt として残す

% --- MAC が読めなかった HE パケットを BSS Color で拾うか --------------------
%     高い MCS (1024QAM 等) では受信SNRが少し足りないだけでデータ部の FCS が
%     通らなくなるが、CSI はプリアンブル (HE-LTF) から算出しているので影響を
%     受けない。HE-SIG-A の BSS Color は CRC を通ったうえで読めるため、
%     MAC が読めたパケットから Color -> BSSID を学習しておけば、読めなかった
%     パケットも同じ BSS のものとして CSI を残せる。
%     ※ Color は 6bit しかないため、近隣に同じ Color の BSS があると混ざる
%       可能性がある。厳密に MAC で確認できたものだけが欲しい場合は false に
%       する (その場合 fcsVerified が false の HE 記録は出力されない)。
useBSSColorFallback = true;

% --- 一括処理から呼ばれたときの上書き ------------------------------------
% decodeIQ_HE_batch.m は、このスクリプトを 1 ファイルずつ呼び出す。
% 冒頭の clear で消えないよう、入力ファイルや保存先は MATLAB 全体の
% 共有領域 (appdata) で受け渡している。単体で実行したときは何も起きない。
batchOutFileName = '';
batchOverride = getappdata(0, 'decodeIQ_HE_batch');
if ~isempty(batchOverride)
    inputRawFile     = batchOverride.inputRawFile;
    hddInputPath     = batchOverride.hddInputPath;
    hddSavePath      = batchOverride.hddSavePath;
    usbSavePath      = batchOverride.usbSavePath;
    targetSSID       = batchOverride.targetSSID;
    batchOutFileName = batchOverride.outFileName;
    showEachPacket   = batchOverride.showEachPacket;
    fprintf('[一括処理] 入力: %s\n', inputRawFile);
end
clear batchOverride;

%% ------------------------------------------------------------------------
%  2. 生IQファイルの読み込み
%  ------------------------------------------------------------------------
% 入力は2種類を受け付ける。
%   *_raw.mat … captureIQ.m の出力。complex double の iq をまるごと読む。
%   *_raw.bin … captureIQ_single.m の出力。int16 のべた書きストリームで、
%               取得条件は同名の *_rawmeta.mat にある。
%               こちらは必要なところだけを少しずつ読むので、キャプチャが
%               何分あってもメモリ使用量は一定に保たれる。分割された
%               *_segNN_raw.mat を作る必要はない。
if isempty(inputRawFile)
    listing = [dir(fullfile(hddInputPath, '*_raw.bin')); ...
               dir(fullfile(hddInputPath, '*_raw.mat'))];
    % 分割セグメントは 1 ファイルの .bin があれば不要なので候補から外す
    listing = listing(~contains({listing.name}, '_seg'));
    if isempty(listing)
        error('decodeIQ_HE:noInputFile', ...
            ['入力ファイルが見つかりません: %s\\*_raw.bin / *_raw.mat\n', ...
             '先に captureIQ_single.m か captureIQ.m を実行するか、\n', ...
             'inputRawFile にパスを指定してください。'], hddInputPath);
    end
    [~, newest] = max([listing.datenum]);
    inputRawFile = fullfile(listing(newest).folder, listing(newest).name);
    fprintf('最新の生IQファイルを自動選択しました:\n  %s\n', inputRawFile);
elseif isempty(fileparts(inputRawFile))
    % ディレクトリ部分が無い = ファイル名だけの指定なので hddInputPath 配下とみなす
    inputRawFile = fullfile(hddInputPath, inputRawFile);
end

if ~isfile(inputRawFile)
    error('decodeIQ_HE:inputNotFound', '入力ファイルが存在しません: %s', inputRawFile);
end

[~, inName, inExt] = fileparts(inputRawFile);
isBinInput = strcmpi(inExt, '.bin');

if isBinInput
    % --- captureIQ_single.m の .bin (取得条件は *_rawmeta.mat 側) ---
    metaFile = fullfile(fileparts(inputRawFile), ...
        [regexprep(inName, '_raw$', '') '_rawmeta.mat']);
    if ~isfile(metaFile)
        error('decodeIQ_HE:noMetaFile', ...
            ['メタデータファイルが見つかりません: %s\n', ...
             '.bin には取得条件が入っていないため、同時に作られる\n', ...
             '*_rawmeta.mat が同じフォルダに必要です。'], metaFile);
    end
    M = load(metaFile, 'meta');
    if ~isfield(M, 'meta')
        error('decodeIQ_HE:badMetaFile', ...
            'メタデータファイルに変数 meta がありません: %s', metaFile);
    end
    meta = M.meta;
    clear M;

    src = openIQSource(inputRawFile, meta);
    iqTotalSamples = src.totalSamples;
else
    % --- captureIQ.m の .mat (従来どおり全部メモリに載せる) ---
    S = load(inputRawFile, 'iq', 'meta');
    if ~isfield(S, 'iq') || ~isfield(S, 'meta')
        error('decodeIQ_HE:badInputFile', ...
            ['入力ファイルに変数 iq / meta がありません: %s\n', ...
             'captureIQ.m が出力した *_raw.mat を指定してください。'], inputRawFile);
    end

    % captureIQ.m は complex double で保存するため通常は変換不要だが、
    % 古い single 形式のファイルも読めるよう double() を通しておく
    % (WLAN Toolbox の関数群は double を前提とする)
    meta = S.meta;
    src  = openIQSource(double(S.iq(:)), meta);
    clear S;
    iqTotalSamples = src.totalSamples;
end

% 以降の処理・保存メタデータはキャプチャ時の条件を引き継ぐ
centerFrequency = meta.centerFrequency;
sampleRate      = meta.sampleRate;
gain            = meta.gain;
captureDuration = meta.captureDuration;
usrpPlatform    = meta.platform;
usrpSerialNum   = meta.serialNum;
overrunCount    = meta.overrunCount;
timestamp       = meta.captureDatetime;

% --- 出力ファイル名 ---
% captureIQ*.m の saveFileName で名前を付けていればその名前 (例: D1_E_antenna1)、
% 無ければキャプチャ時刻を頭に付ける
ssidSafe = regexprep(targetSSID, '[^A-Za-z0-9_-]', '_');
outBase  = timestamp;
if isfield(meta, 'saveName') && ~isempty(meta.saveName)
    outBase = char(meta.saveName);
end
outFileName = [outBase '_' ssidSafe '_CSI.mat'];
if ~isempty(batchOutFileName)
    % 一括処理では入力ファイル名から出力名を決める (呼び出し側で決定済み)。
    % meta.captureDatetime は分割セグメント間で同じ値になるため、そのままだと
    % 全セグメントが同じ名前で上書きし合ってしまう。
    outFileName = batchOutFileName;
end

% --- 画面に表示する内容をテキストファイルにも残す ---
% 出力 (*_CSI.mat) と同じフォルダに、同じ名前で *_CSI_log.txt を作る。
% 生IQ ファイルごとに 1 つできるので、あとで復号結果を見比べられる。
% すでに diary が動いている場合 (一括処理のログなど) は、終わったら元に戻す。
decodeLog = struct('file', '', 'prevFile', '', 'prevOn', false);
if saveDecodeLog
    decodeLog = startDecodeLog(fullfile(hddSavePath, ...
        regexprep(outFileName, '\.mat$', '_log.txt')));
end

fprintf('\n読み込んだ生IQ:\n');
fprintf('  ファイル        : %s\n', inputRawFile);
fprintf('  キャプチャ日時  : %s\n', timestamp);
fprintf('  サンプル数      : %d (%.3f s)\n', iqTotalSamples, iqTotalSamples / sampleRate);
fprintf('  中心周波数      : %.4f GHz\n', centerFrequency / 1e9);
fprintf('  サンプルレート  : %.3f MSps\n', sampleRate / 1e6);
fprintf('  ゲイン          : %d dB\n', gain);
fprintf('  オーバーラン    : %d 回\n', overrunCount);
if overrunCount > 0
    fprintf('  ※キャプチャ時に取りこぼしがあります。時刻軸に不連続が含まれる\n');
    fprintf('    可能性があるため、CSI の時系列解釈には注意してください。\n');
end
fprintf('  対象 SSID       : %s\n', targetSSID);

% --- HE 復号に必要な関数が揃っているかを先に確認する ---------------------
% 途中でいきなり "Undefined function" になるより、開始前に分かるほうがよい。
heFuncs = {'wlanHEDemodulate', 'wlanHEOFDMInfo', 'wlanHEEqualize', ...
           'wlanHESIGABitRecover', 'wlanLSIGBitRecover', ...
           'wlanHERecoveryConfig', 'wlanHELTFChannelEstimate', ...
           'wlanHEDataBitRecover', 'wlanHESUConfig'};
heAvailable = true;
missingFuncs = {};
for k = 1:numel(heFuncs)
    if isempty(which(heFuncs{k}))
        heAvailable = false;
        missingFuncs{end+1} = heFuncs{k}; %#ok<SAGROW>
    end
end
if heAvailable
    fprintf('  HE 復号         : 有効\n');
else
    warning('decodeIQ_HE:noHESupport', ...
        ['この MATLAB では HE の復号ができません (WLAN Toolbox が古いか未導入)。\n', ...
         '  不足している関数: %s\n', ...
         '  Non-HT / HT / VHT のみ復号して続行します。'], strjoin(missingFuncs, ', '));
end

% --- 受信レベルの確認 ------------------------------------------------------
% ADC が飽和していると、BPSK/QPSK のプリアンブルや SIG フィールドは通るのに
% 高次QAM (64QAM以上) のデータ部だけが壊れる、という症状が出る。HE は
% MCS8〜11 で 1024QAM まで使うため、ここの確認が効く。
% 全サンプルを評価するとメモリを大量に使うので、間引いて概算する。
% 全部読むと長時間キャプチャでは時間もメモリもかかるので、キャプチャ全体に
% 等間隔で散らした小さなブロックだけを見て概算する。
nProbe      = 20;
probeLen    = min(250000, iqTotalSamples);
levelSample = complex(zeros(nProbe * probeLen, 1));
nFilled     = 0;
for pk = 1:nProbe
    pos = round((pk - 1) / nProbe * max(iqTotalSamples - probeLen, 0));
    chunk = readIQ(src, pos, probeLen);
    levelSample(nFilled + (1:numel(chunk))) = chunk;
    nFilled = nFilled + numel(chunk);
end
levelSample = levelSample(1:nFilled);
peakLevel   = max(max(abs(real(levelSample))), max(abs(imag(levelSample))));
rmsLevel    = sqrt(mean(abs(levelSample).^2));
satRatio    = mean(abs(real(levelSample)) > 0.99 * peakLevel | ...
                   abs(imag(levelSample)) > 0.99 * peakLevel);
clear levelSample;
fprintf('  受信レベル      : ピーク %.4f, RMS %.5f (PAPR %.1f dB)\n', ...
    peakLevel, rmsLevel, 20*log10(peakLevel / max(rmsLevel, eps)));
if satRatio > 1e-4
    fprintf(['  ※サンプルの %.3f%% がピーク付近に張り付いています。ADC が飽和\n', ...
             '    している可能性が高いです。飽和すると BPSK/QPSK は通るのに\n', ...
             '    64QAM 以上だけが壊れるため、captureIQ.m の gain を 10 dB 程度\n', ...
             '    下げて取り直してください。\n'], satRatio * 100);
end

%% ------------------------------------------------------------------------
%  3. 出力先の準備 (HDD と USB メモリの両方)
%  ------------------------------------------------------------------------
% (出力ファイル名 outFileName は第2節で決定済み)
outMatFiles = {};   % 実際に書き込めた出力先
outTargets  = { 'HDD', hddSavePath; 'USB', usbSavePath };
for k = 1:size(outTargets, 1)
    label = outTargets{k, 1};
    pth   = outTargets{k, 2};

    if isempty(pth)
        % 未設定 (例: USB メモリを使わない) のでこの出力先はスキップ
        continue;
    end

    if ~exist(pth, 'dir')
        fprintf('%s 保存先フォルダが存在しないため作成します: %s\n', label, pth);
        [ok, msg] = mkdir(pth);
        if ~ok
            warning('decodeIQ_HE:mkdirFailed', ...
                '%s 保存先フォルダを作成できませんでした (%s): %s', label, pth, msg);
            continue;
        end
    end

    testFile = fullfile(pth, '.write_test.tmp');
    fidTest  = fopen(testFile, 'w');
    if fidTest == -1
        warning('decodeIQ_HE:notWritable', ...
            '%s 保存先に書き込みできないためスキップします: %s', label, pth);
        continue;
    end
    fclose(fidTest);
    delete(testFile);

    outMatFiles{end+1} = fullfile(pth, outFileName); %#ok<SAGROW>
end

if isempty(outMatFiles)
    error('decodeIQ_HE:noWritableOutput', ...
        ['書き込み可能な出力先がありません。\n', ...
         '  HDD: %s\n  USB: %s\n', ...
         'ドライブの接続とパスを確認してください。'], hddSavePath, usbSavePath);
end

fprintf('  保存先          :\n');
for k = 1:numel(outMatFiles)
    fprintf('    %s\n', outMatFiles{k});
end

%% ------------------------------------------------------------------------
%  4. パケット検出 → CSI 推定 → MAC 復号
%  ------------------------------------------------------------------------
subcarrierIndicesNonHT = [-26:-1, 1:26];    % Non-HT CBW20 (52本)
subcarrierIndicesHT20  = [-26:-1, 1:26];    % HT     CBW20 (52本, Non-HTと同じ配置)
subcarrierIndicesVHT20 = [-28:-1, 1:28];    % VHT    CBW20 (56本)
% HE は 20MHz でも 256点FFT (78.125kHz間隔) の 242-tone RU を使うため
% サブキャリア番号の意味が他フォーマットと異なる。既定値は規格上の
% 242-tone RU の配置で、実際には最初に復号できた HE パケットの
% wlanHEOFDMInfo('HE-Data', ...) から取得した値で上書きする。
subcarrierIndicesHE20  = [-122:-2, 2:122];  % HE     CBW20 (242本)

pktLog = struct('timeSec', {}, 'bssid', {}, 'addr1', {}, 'frameType', {}, 'ssid', {}, ...
    'phyFormat', {}, 'bssColor', {}, 'uplink', {}, 'fcsVerified', {}, 'mpduCount', {}, 'csi', {});
bssidToSSID = containers.Map('KeyType', 'char', 'ValueType', 'char');

% 復号統計 (診断用)
stats = struct('detected', 0, 'timingSkip', 0, 'nonHT', 0, 'ht', 0, 'vht', 0, ...
    'he', 0, 'htGF', 0, 'other', 0, 'htUnsupported', 0, 'vhtUnsupported', 0, ...
    'heUnsupported', 0, 'decodeOK', 0, 'csiOnly', 0, 'noAddr2', 0, 'errors', 0);
errMsgs       = containers.Map('KeyType', 'char', 'ValueType', 'double');
deagStatusCnt = containers.Map('KeyType', 'char', 'ValueType', 'double');
nonBinaryShown = false;  % 非0/1データを検出した旨を一度だけ表示するためのフラグ
htRejects     = containers.Map('KeyType', 'char', 'ValueType', 'double');
htMCSCounts   = containers.Map('KeyType', 'double', 'ValueType', 'double');
htReservedOK  = 0;   % HT-SIG の予約ビットが仕様どおりだった件数
htSigParsed   = 0;   % HT-SIG の CRC を通り、ビット解釈まで到達した件数
vhtSigParsed  = 0;   % VHT-SIG-A の CRC を通り、ビット解釈まで到達した件数
vhtRejects    = containers.Map('KeyType', 'char', 'ValueType', 'double');
vhtBWCounts   = containers.Map('KeyType', 'double', 'ValueType', 'double');
vhtNSTSCounts = containers.Map('KeyType', 'double', 'ValueType', 'double');
vhtReservedOK = 0;   % VHT-SIG-A の予約ビットが仕様どおりだった件数
vhtDetailShown = 0;  % 内訳を表示した VHT パケット数
vhtDetailMax   = 5;  % 内訳を表示する最大件数

% --- HE 用の集計 ---
heFormatCounts = containers.Map('KeyType', 'char', 'ValueType', 'double');
heRejects      = containers.Map('KeyType', 'char', 'ValueType', 'double');
heMCSCounts    = containers.Map('KeyType', 'double', 'ValueType', 'double');
heNSTSCounts   = containers.Map('KeyType', 'double', 'ValueType', 'double');
heBWCounts     = containers.Map('KeyType', 'double', 'ValueType', 'double');
heCodingCounts = containers.Map('KeyType', 'char', 'ValueType', 'double');
heBSSColors    = containers.Map('KeyType', 'double', 'ValueType', 'double');
heMCSOKCounts  = containers.Map('KeyType', 'double', 'ValueType', 'double');
heSNRs         = [];  % HE パケットごとの受信SNR推定 [dB]
heMPDUStatus   = containers.Map('KeyType', 'char', 'ValueType', 'double');
heHdrReasons   = containers.Map('KeyType', 'char', 'ValueType', 'double');
heMPDUBytes    = [];  % 取り出せた MPDU の長さ [byte]
hePilotTracked = 0;   % HE-Data の位相追跡が実際に効いた件数
hePilotTotal   = 0;   % HE-Data の復号まで到達した件数
heSigParsed    = 0;   % HE-SIG-A の CRC と解釈を通った件数
heDetailShown  = 0;   % 内訳を表示した HE パケット数
heDetailMax    = 5;   % 内訳を表示する最大件数
heSubcResolved = false;  % subcarrierIndicesHE20 を実測値で上書き済みか

minPreambleLen = 560;   % L-STF..L-SIG + 2シンボル (レガシーのフォーマット検出まで)
searchOffset = 0;

% 1パケットの復号に切り出す最大長。
%   802.11 は 1 つの PPDU の継続時間を 5.484 ms 以下と定めている
%   (aPPDUMaxTime)。20 MSps なら約 11 万サンプルなので、6 ms 分あれば
%   どんなパケットでも全体が収まる。
%
%   ここを区切るのは速度のためで、効果は非常に大きい。区切らずに
%   iq(pktStart+1:end) を渡すと、パケット 1 個ごとに「残りのIQ全部」
%   (5秒キャプチャなら 1 億サンプル = 1.6 GB) をコピーして、その全要素に
%   CFO補正の複素指数を掛ける処理が 2 回走る。実際に使うのは先頭の
%   数万サンプルだけなので、大半が完全な無駄になる。
maxPPDULen = round(6e-3 * sampleRate);

fprintf('\nオフライン復号を開始します...\n');
decodeTic = tic;

% 検出は窓に区切って行う (毎回バッファ全体を走査すると非常に遅くなるため)。
% 窓の境界にまたがるプリアンブルを取りこぼさないよう少し重ねて進める。
detectWindow  = 400000;   % 20 ms 分
detectOverlap = 512;      % プリアンブル検出に必要な重なり

% --- ブロック単位でメモリに載せる ---------------------------------------
% 入力が .bin のときは、ここで指定した分だけをその都度ファイルから読む。
% キャプチャが 5 秒でも 10 分でも、メモリ使用量はこのブロック1個分で
% 頭打ちになる。だから長時間キャプチャをセグメント分割する必要がない。
% 末尾に 1 PPDU 分の重なりを付けて、境界にまたがるパケットを拾う。
blockSamples = 20e6;      % 1 ブロック = 1 秒分 (complex double で約 320 MB)
blockStart   = 0;         % このブロック先頭のグローバルサンプル位置 (0-based)

while blockStart < iqTotalSamples
    iq = readIQ(src, blockStart, blockSamples + maxPPDULen);
    if numel(iq) < minPreambleLen
        break;
    end
    isLastBlock = (blockStart + numel(iq) >= iqTotalSamples);
    % このブロックが担当するのは先頭 blockSamples 分だけ。
    % それ以降に始まるパケットは次のブロックで扱う (二重計上の防止)。
    if isLastBlock
        blockLimit = numel(iq);
    else
        blockLimit = blockSamples;
    end
    searchOffset = 0;

% ここから下はブロック内の走査。iq / searchOffset / pktOffset / pktStart は
% すべてブロック先頭からの相対位置で、絶対時刻は blockStart を足して求める。
while searchOffset + minPreambleLen <= numel(iq) && searchOffset < blockLimit
    % --- パケット検出 (窓内の相対位置が返る) ---
    winEnd    = min(numel(iq), searchOffset + detectWindow);
    relOffset = wlanPacketDetect(iq(searchOffset+1 : winEnd), chanBW, 0, pktDetThreshold);
    if isempty(relOffset)
        if winEnd >= numel(iq)
            break;   % 末尾まで探索済み
        end
        searchOffset = winEnd - detectOverlap;
        continue;
    end
    pktOffset = searchOffset + relOffset;   % 0-based (ブロック内の位置)

    if ~isLastBlock && pktOffset >= blockLimit
        break;   % このブロックの担当範囲を超えた。次のブロックで扱う
    end
    if numel(iq) - pktOffset < minPreambleLen
        break;   % 末尾が足りない
    end

    stats.detected = stats.detected + 1;

    % 次回の検索開始位置 (最低でも L-STF 分は進める。成功時は後で上書き)
    nextSearch = pktOffset + 160;

    try
        % --- 粗CFO推定 (L-STF) ---
        coarseCFO = wlanCoarseCFOEstimate(iq(pktOffset + (1:160)), chanBW);

        % --- 精タイミング推定 (粗CFO補正した Non-HT プリアンブル) ---
        nonHTPre  = applyCFO(iq(pktOffset + (1:400)), sampleRate, -coarseCFO);
        symOffset = wlanSymbolTimingEstimate(nonHTPre, chanBW);

        % wlanSymbolTimingEstimate は負値を返し得る。真の先頭が
        % バッファ範囲外になる場合は誤検出として読み捨てる。
        pktStart = pktOffset + symOffset;    % 0-based
        if pktStart < 0 || numel(iq) - pktStart < minPreambleLen
            stats.timingSkip = stats.timingSkip + 1;
            searchOffset = nextSearch;
            continue;
        end

        % --- 精CFO推定・補正 (L-LTF) ---
        %     1 PPDU 分 (最大 maxPPDULen サンプル) だけを切り出して補正する。
        %     この窓に収まりきらない = 規格上の最大PPDU長を超えるパケットは
        %     復号側で「サンプル不足」と判定され読み捨てられるが、それは
        %     誤検出なので問題ない。
        pktEnd  = min(numel(iq), pktStart + maxPPDULen);
        pkt     = applyCFO(iq(pktStart+1:pktEnd), sampleRate, -coarseCFO);
        fineCFO = wlanFineCFOEstimate(pkt(161:320), chanBW);
        pkt     = applyCFO(pkt, sampleRate, -fineCFO);

        % --- L-LTF を復調してからチャネル推定・ノイズ推定 ---
        %     (wlanLLTFChannelEstimate は時間領域波形ではなく復調済み
        %      シンボルを受け取る点に注意)
        lltfDemod   = wlanLLTFDemodulate(pkt(161:320), chanBW);
        lltfChanEst = wlanLLTFChannelEstimate(lltfDemod, chanBW);
        noiseEst    = wlanLLTFNoiseEstimate(lltfDemod);

        % --- フォーマット検出 ---
        %     wlanFormatDetect は L-LTF の後ろの OFDM シンボルを見て判定する。
        %     3シンボル (L-SIG + 2) までならレガシー(Non-HT/HT/VHT)判定、
        %     4シンボル (L-SIG + RL-SIG + HE-SIG-A x2) 渡すと HE も判定できる。
        %     末尾で 4 シンボル取れないときは 3 シンボルで呼び、
        %     レガシー判定だけを行う。
        %     内部の L-SIG チェック失敗警告は独自サマリで集計するため抑止する
        nSymAvail = floor((numel(pkt) - 320) / 80);
        nSymUse   = min(4, nSymAvail);
        if nSymUse < 3
            stats.timingSkip = stats.timingSkip + 1;
            searchOffset = nextSearch;
            continue;
        end
        wState = warning('off', 'all');
        fmt = wlanFormatDetect(pkt(321 : 320 + nSymUse * 80), ...
            lltfChanEst, noiseEst, chanBW);
        warning(wState);
        fmtStr = upper(string(fmt));

        switch true
            case fmtStr == "NON-HT"
                stats.nonHT = stats.nonHT + 1;
                [st, consumed, entry] = processNonHT(pkt, lltfChanEst, noiseEst, chanBW);

            case fmtStr == "HT-MF"
                stats.ht = stats.ht + 1;
                [st, consumed, entry, htInfo] = processHT(pkt, lltfChanEst, noiseEst, chanBW);

                if htInfo.mcs >= 0
                    htSigParsed = htSigParsed + 1;
                    htMCSCounts = bumpMap(htMCSCounts, htInfo.mcs);
                end
                if htInfo.reservedOK
                    htReservedOK = htReservedOK + 1;
                end
                [deagStatusCnt, nonBinaryShown] = ...
                    recordDeag(deagStatusCnt, nonBinaryShown, 'HT', htInfo);
                if ~isempty(htInfo.reason)
                    stats.htUnsupported = stats.htUnsupported + 1;
                    htRejects = bumpMap(htRejects, htInfo.reason);
                end

            case fmtStr == "VHT"
                stats.vht = stats.vht + 1;
                [st, consumed, entry, vhtInfo] = processVHT(pkt, lltfChanEst, noiseEst, chanBW);

                if vhtInfo.bwMHz > 0
                    vhtSigParsed = vhtSigParsed + 1;
                    vhtBWCounts = bumpMap(vhtBWCounts, vhtInfo.bwMHz);
                end
                if vhtInfo.nsts > 0
                    vhtNSTSCounts = bumpMap(vhtNSTSCounts, vhtInfo.nsts);
                end
                if vhtInfo.reservedOK
                    vhtReservedOK = vhtReservedOK + 1;
                end
                [deagStatusCnt, nonBinaryShown] = ...
                    recordDeag(deagStatusCnt, nonBinaryShown, 'VHT', vhtInfo);
                % 最初の数件だけ内訳を表示 (パケット長算出が妥当か確認するため)
                if vhtDetailShown < vhtDetailMax && vhtInfo.nsts == 1 && vhtInfo.bwMHz == 20
                    vhtDetailShown = vhtDetailShown + 1;
                    fprintf(['    <VHT#%d> MCS=%d GI=%s 符号化=%s | L-SIG長=%d NSYM=%d\n', ...
                             '            APEP: L-SIG由来=%d SIG-B由来=%d 採用=%d | ', ...
                             'A-MPDU分解=%s MPDU数=%d\n'], ...
                        vhtDetailShown, vhtInfo.mcs, ternary(vhtInfo.gi==1,'Short','Long'), ...
                        ternary(vhtInfo.coding==1,'LDPC','BCC'), vhtInfo.lsigLen, vhtInfo.nsym, ...
                        vhtInfo.apepFromLSIG, vhtInfo.apepFromSIGB, vhtInfo.apepUsed, ...
                        vhtInfo.deagStatus, vhtInfo.mpduCount);
                end
                if ~isempty(vhtInfo.reason)
                    stats.vhtUnsupported = stats.vhtUnsupported + 1;
                    vhtRejects = bumpMap(vhtRejects, vhtInfo.reason);
                end

            case startsWith(fmtStr, "HE")
                stats.he = stats.he + 1;
                heFormatCounts = bumpMap(heFormatCounts, char(fmtStr));

                if ~heAvailable
                    % WLAN Toolbox が HE 非対応。検出だけ記録して読み捨てる
                    stats.heUnsupported = stats.heUnsupported + 1;
                    heRejects = bumpMap(heRejects, 'WLAN Toolbox が HE 未対応');
                    st = 0; consumed = 640; entry = [];
                else
                    [st, consumed, entry, heInfo] = ...
                        processHE(pkt, lltfChanEst, noiseEst, chanBW, fmtStr);

                    if heInfo.sigaParsed
                        heSigParsed  = heSigParsed + 1;
                        % 受信SNRの概算 (L-LTF のチャネル推定電力 / 雑音分散)。
                        % どのMCSまで復号できる見込みがあるかの判断に使う。
                        heSNRs(end+1) = 10 * log10( ...
                            mean(abs(lltfChanEst(:)).^2) / max(noiseEst, eps)); %#ok<SAGROW>
                        heMCSCounts  = bumpMap(heMCSCounts, heInfo.mcs);
                        heNSTSCounts = bumpMap(heNSTSCounts, heInfo.nsts);
                        heBWCounts   = bumpMap(heBWCounts, heInfo.bwMHz);
                        if ~isempty(heInfo.coding)
                            heCodingCounts = bumpMap(heCodingCounts, heInfo.coding);
                        end
                        if heInfo.bssColor >= 0
                            heBSSColors = bumpMap(heBSSColors, heInfo.bssColor);
                        end
                    end
                    [deagStatusCnt, nonBinaryShown] = ...
                        recordDeag(deagStatusCnt, nonBinaryShown, 'HE', heInfo);

                    % 「なぜ MPDU が読めなかったか」の切り分け用の集計
                    for q = 1:numel(heInfo.mpduStatus)
                        heMPDUStatus = bumpMap(heMPDUStatus, heInfo.mpduStatus{q});
                    end
                    for q = 1:numel(heInfo.hdrReason)
                        heHdrReasons = bumpMap(heHdrReasons, heInfo.hdrReason{q});
                    end
                    if ~isempty(heInfo.mpduBytes)
                        heMPDUBytes = [heMPDUBytes, heInfo.mpduBytes(:).']; %#ok<AGROW>
                    end

                    % HE のサブキャリア番号は復号できたパケットから実測する
                    if ~heSubcResolved && ~isempty(heInfo.subcIdx)
                        subcarrierIndicesHE20 = heInfo.subcIdx(:).';
                        heSubcResolved = true;
                    end

                    % 最初の数件だけ内訳を表示 (パラメータ推定が妥当かの確認用)
                    if heDetailShown < heDetailMax && heInfo.sigaParsed
                        heDetailShown = heDetailShown + 1;
                        fprintf(['    <HE#%d> %s MCS=%d NSTS=%d BW=%dMHz GI=%.1fus ', ...
                                 'HE-LTF=%dx 符号化=%s DCM=%d\n', ...
                                 '           BSSColor=%d %s | L-SIG長=%d PSDU長=%dB ', ...
                                 'サブキャリア=%d本 | A-MPDU分解=%s MPDU数=%d\n'], ...
                            heDetailShown, heInfo.packetFormat, heInfo.mcs, heInfo.nsts, ...
                            heInfo.bwMHz, heInfo.giUs, heInfo.heltfType, heInfo.coding, ...
                            heInfo.dcm, heInfo.bssColor, ...
                            ternary(heInfo.uplink, '上り', '下り'), ...
                            heInfo.lsigLen, heInfo.psduLen, heInfo.numSubcarriers, ...
                            heInfo.deagStatus, heInfo.mpduCount);
                    end

                    % HE-Data の復号まで到達したものについて、位相追跡が
                    % 実際に効いたかを数える (棄却されたものも含めて数える)
                    if heInfo.numSubcarriers > 0
                        hePilotTotal = hePilotTotal + 1;
                        if heInfo.pilotTracked
                            hePilotTracked = hePilotTracked + 1;
                        end
                    end

                    if ~isempty(heInfo.reason)
                        stats.heUnsupported = stats.heUnsupported + 1;
                        heRejects = bumpMap(heRejects, heInfo.reason);
                    elseif st > 0 && heInfo.sigaParsed
                        % MAC まで到達できたものだけを MCS 別に数える。
                        % 上の「MCS の内訳」(SIG-A が読めた全件) と見比べると、
                        % どの変調方式で復号が破綻しているかが分かる。
                        heMCSOKCounts = bumpMap(heMCSOKCounts, heInfo.mcs);
                    end
                end

            case fmtStr == "HT-GF"
                % Greenfield はプリアンブル構造自体が異なり非対応
                stats.htGF = stats.htGF + 1;
                st = 0; consumed = 320; entry = [];

            otherwise
                stats.other = stats.other + 1;
                st = 0; consumed = 320; entry = [];
        end

        if st < 0
            % パケット後半がキャプチャ範囲外 (末尾で切れている)
            searchOffset = nextSearch;
            continue;
        end

        if st == 2
            % MAC は読めなかったが CSI は取れている HE パケット。
            % 後段で BSS Color を手掛かりに対象SSIDへ帰属させる。
            stats.csiOnly = stats.csiOnly + 1;
            entry.timeSec = (blockStart + pktStart) / sampleRate;
            pktLog(end+1) = entry; %#ok<SAGROW>
        elseif st > 0
            stats.decodeOK = stats.decodeOK + 1;

            if isempty(entry)
                % ACK/CTS など Address2 を持たないフレーム (BSSID 判定に使えない)
                stats.noAddr2 = stats.noAddr2 + 1;
            else
                entry.timeSec = (blockStart + pktStart) / sampleRate;
                pktLog(end+1) = entry; %#ok<SAGROW>

                if any(strcmpi(entry.frameType, {'Beacon', 'Probe Response'})) && ~isempty(entry.ssid)
                    bssidToSSID(entry.bssid) = entry.ssid;
                end

                if showEachPacket
                    fprintf('  [%7.4fs] %-6s %-16s BSSID=%s%s%s\n', entry.timeSec, ...
                        entry.phyFormat, entry.frameType, entry.bssid, ...
                        ternary(~isempty(entry.ssid), sprintf('  SSID="%s"', entry.ssid), ''), ...
                        ternary(entry.mpduCount > 1, sprintf('  (MPDU集約x%d)', entry.mpduCount), ''));
                end
            end
        end

        searchOffset = pktStart + max(consumed, 160);

    catch ME
        stats.errors = stats.errors + 1;
        key = ME.message;
        if isKey(errMsgs, key)
            errMsgs(key) = errMsgs(key) + 1;
        else
            errMsgs(key) = 1;
        end
        if verboseErrors
            warning('decodeIQ_HE:decodeError', 'パケット処理中にエラー: %s', ME.message);
        end
        searchOffset = nextSearch;
    end
end

    blockStart = blockStart + blockSamples;   % 次のブロックへ
end   % ブロックループの終わり

closeIQSource(src);
clear iq;

elapsedDecode = toc(decodeTic);

%% ------------------------------------------------------------------------
%  5. 復号結果のサマリ表示
%  ------------------------------------------------------------------------
fprintf('\n[復号サマリ] (所要 %.2f s)\n', elapsedDecode);
fprintf('  パケット検出数            : %d\n', stats.detected);
fprintf('    タイミング不正でスキップ: %d\n', stats.timingSkip);
fprintf('    Non-HT として検出       : %d\n', stats.nonHT);
fprintf('    HT     として検出       : %d  (うち非対応構成 %d)\n', stats.ht, stats.htUnsupported);
fprintf('    VHT    として検出       : %d  (うち非対応構成 %d)\n', stats.vht, stats.vhtUnsupported);
fprintf('    HE     として検出       : %d  (うち非対応構成 %d)\n', stats.he, stats.heUnsupported);
fprintf('    HT-Greenfield            : %d  (非対応)\n', stats.htGF);
fprintf('    その他フォーマット      : %d  (非対応)\n', stats.other);

if stats.ht > 0
    fprintf('  [HT 詳細]\n');
    printCountMapNum(htMCSCounts, '    MCS の内訳: ', 'MCS%d=%d回  ');
    printCountMapChar(htRejects, '    非対応の理由: ');
    % 分母は「HT-SIG の CRC を通り、ビット解釈まで到達した件数」。検出だけ
    % されて CRC で落ちたものを含めると、自前パースの妥当性を測れない。
    fprintf('    HT-SIG予約ビット検証: %d/%d 個が仕様どおり (CRC通過分のみ)', ...
        htReservedOK, htSigParsed);
    if htSigParsed > 0 && htReservedOK < htSigParsed * 0.5
        fprintf('  <-- 大半が不一致。ビット解釈がズレている可能性あり\n');
    else
        fprintf('  (ビット解釈は妥当)\n');
    end
end

if stats.vht > 0
    fprintf('  [VHT 詳細]\n');
    printCountMapNum(vhtBWCounts, '    送信帯域幅の内訳: ', '%dMHz=%d回  ');
    if vhtNSTSCounts.Count > 0
        nKeys = cell2mat(keys(vhtNSTSCounts));
        fprintf('    空間ストリーム数: ');
        for k = 1:numel(nKeys)
            fprintf('NSTS=%d が %d回  ', nKeys(k), vhtNSTSCounts(nKeys(k)));
        end
        fprintf('\n');
        multiKeys = nKeys(nKeys >= 2);
        if ~isempty(multiKeys)
            nMultiStream = sum(cellfun(@(k) vhtNSTSCounts(k), num2cell(multiKeys)));
            fprintf(['    ※NSTS>=2 の分 (計%d件) は送信元 BSSID が分からず、\n', ...
                     '      特定のSSIDに帰属させることができません。\n'], nMultiStream);
        end
    end
    printCountMapChar(vhtRejects, '    非対応の理由    : ');
    % VHT-SIG-A のビット解釈が正しいかの自己判定
    fprintf('    SIG-A予約ビット検証: %d/%d 個が仕様どおり (CRC通過分のみ)', ...
        vhtReservedOK, vhtSigParsed);
    if vhtSigParsed > 0 && vhtReservedOK < vhtSigParsed * 0.5
        fprintf('  <-- 大半が不一致。ビット解釈がズレている可能性あり\n');
    else
        fprintf('  (ビット解釈は妥当)\n');
    end
end

if stats.he > 0
    fprintf('  [HE 詳細]\n');
    printCountMapChar(heFormatCounts, '    PPDU形式の内訳  : ');
    fprintf('    HE-SIG-A 解釈成功: %d 件 (検出 %d 件中)\n', heSigParsed, stats.he);
    printCountMapNum(heBWCounts,   '    送信帯域幅の内訳: ', '%dMHz=%d回  ');
    printCountMapNum(heMCSCounts,  '    MCS の内訳      : ', 'MCS%d=%d回  ');
    printCountMapNum(heMCSOKCounts, '    うち復号成功    : ', 'MCS%d=%d回  ');
    if heMCSCounts.Count > 0 && heMCSOKCounts.Count == 0
        fprintf(['    ※SIG-A は読めているのに MAC まで到達したものが 0 件です。\n', ...
                 '      プリアンブルは復号できているので受信自体は成立しており、\n', ...
                 '      データ部の変調が受信品質に対して重すぎる (高MCS)、\n', ...
                 '      または ADC が飽和している可能性が高いです。\n']);
    end
    printCountMapChar(heCodingCounts, '    符号化方式      : ');
    if heNSTSCounts.Count > 0
        nKeys = cell2mat(keys(heNSTSCounts));
        fprintf('    空間ストリーム数: ');
        for k = 1:numel(nKeys)
            fprintf('NSTS=%d が %d回  ', nKeys(k), heNSTSCounts(nKeys(k)));
        end
        fprintf('\n');
        multiKeys = nKeys(nKeys >= 2);
        if ~isempty(multiKeys)
            nMultiStream = sum(cellfun(@(k) heNSTSCounts(k), num2cell(multiKeys)));
            fprintf(['    ※NSTS>=2 の分 (計%d件) は受信アンテナ1本では復号できず、\n', ...
                     '      特定のSSIDに帰属させることができません。\n'], nMultiStream);
        end
    end
    % BSS Color は HE-SIG-A に平文で入っており、CRC を通っていれば信頼できる。
    % 同じ BSS のパケットかどうかの二次的な手掛かりになる。
    printCountMapNum(heBSSColors, '    BSS Color       : ', 'Color%d=%d回  ');
    printCountMapChar(heRejects,  '    非対応の理由    : ');

    % --- MPDU が読めなかった理由の内訳 ---------------------------------
    % ここが切り分けの核心。
    %   FCSFailed が大半      -> ペイロードにビット誤り。受信品質の問題。
    %   それ以外が大半        -> ビットは正しいのに MATLAB が解釈できて
    %                            いない (非対応のフレーム種別など)。
    %                            この場合は復号器側で拾える。
    printCountMapChar(heMPDUStatus, '    MPDU復号ステータス: ');
    % 自前のヘッダ解析がどこで弾いたか。ビット誤りなら ProtocolVersion≠0 や
    % Address2不正が多くなる。長さ不足が多いなら A-MPDU の切り出しがおかしい。
    printCountMapChar(heHdrReasons,  '    MACヘッダ棄却理由 : ');
    if ~isempty(heMPDUBytes)
        edges = [0 24 100 500 1600 inf];
        lbl   = {'<24B', '24-99B', '100-499B', '500-1599B', '>=1600B'};
        fprintf('    MPDU長の分布      : ');
        for q = 1:numel(lbl)
            fprintf('%s=%d件  ', lbl{q}, ...
                sum(heMPDUBytes >= edges(q) & heMPDUBytes < edges(q+1)));
        end
        fprintf('(中央値 %d B)\n', round(median(heMPDUBytes)));
    end

    % --- 位相追跡が実際に効いたか ---
    % ここが 0 だと、A-MPDU の後半が位相回転で壊れたままになる。
    if hePilotTotal > 0
        fprintf('    位相追跡(HE-Data): %d/%d 件に適用', hePilotTracked, hePilotTotal);
        if hePilotTracked == 0
            fprintf('  <-- 効いていません。上の警告を確認してください\n');
        else
            fprintf('\n');
        end
    end

    % --- 受信SNRと、観測されたMCSの所要SNRの比較 ---
    % これが最終的な判断材料。SNRが所要値に届いていなければ、
    % 復号器を改良しても復号率は上がらない。
    if ~isempty(heSNRs)
        sortedSNR = sort(heSNRs);
        medSNR    = sortedSNR(max(1, round(0.50 * numel(sortedSNR))));
        loSNR     = sortedSNR(max(1, round(0.10 * numel(sortedSNR))));
        hiSNR     = sortedSNR(max(1, round(0.90 * numel(sortedSNR))));
        fprintf('    受信SNR推定     : 中央値 %.1f dB (下位10%% %.1f dB / 上位10%% %.1f dB)\n', ...
            medSNR, loSNR, hiSNR);

        % 最も多かった MCS の所要SNRと比べる
        mcsKeys   = cell2mat(keys(heMCSCounts));
        mcsVals   = cell2mat(values(heMCSCounts));
        [~, iMax] = max(mcsVals);
        domMCS    = mcsKeys(iMax);
        reqSNR    = heRequiredSNR(domMCS);
        fprintf('    最多MCS%d の所要SNR: 約 %d dB (%s)', domMCS, reqSNR, heModulationName(domMCS));
        if medSNR < reqSNR
            fprintf('  <-- %.1f dB 不足\n', reqSNR - medSNR);
            fprintf(['      受信品質に対して変調が重すぎます。復号器側では改善\n', ...
                     '      できないので、送信側の MCS を下げてください。\n', ...
                     '      (PicoScenes で送信するなら MCS を 0〜4 に固定する、\n', ...
                     '       AP が送信側なら iperf3 の向きを逆にして PicoScenes を\n', ...
                     '       送信側にする、アンテナを近づける、等)\n']);
        else
            fprintf('  (SNRは足りている)\n');
        end
    end
    if heSigParsed == 0 && stats.he > 0
        fprintf(['    ※HE-SIG-A を1件も解釈できていません。受信SNR不足か、\n', ...
                 '      HE パケットが 40/80MHz で送信されている可能性があります。\n']);
    end
end

fprintf('  MAC まで復号成功          : %d\n', stats.decodeOK);
fprintf('    うち Address2 無し(ACK/CTS等、BSSID判定不可): %d\n', stats.noAddr2);
fprintf('  CSIのみ記録(HE, MAC未復号): %d  (BSS Color で帰属を試みます)\n', stats.csiOnly);
fprintf('  復号エラー                : %d\n', stats.errors);

% --- A-MPDU 分解の結果 (データ復号が機能しているかの判断材料) ---
if deagStatusCnt.Count > 0
    fprintf('  [A-MPDU 分解結果]\n');
    dKeys = keys(deagStatusCnt);
    for k = 1:numel(dKeys)
        fprintf('    %-40s : %d回\n', dKeys{k}, deagStatusCnt(dKeys{k}));
    end
end

% --- FCS 検証の内訳 (全BSSID) ---
% FCS検証OK が 0 のフォーマットは、そのフォーマットのデータ復号が
% 成立していない (SNR不足かパラメータ推定ミス) ことを意味する。
if ~isempty(pktLog)
    fprintf('  [FCS 検証の内訳 (全BSSID)]\n');
    allFmt  = {pktLog.phyFormat};
    allFcsV = logical([pktLog.fcsVerified]);
    allNoMAC = strcmp({pktLog.frameType}, '(MAC未復号)');
    for f = {'Non-HT', 'HT', 'VHT', 'HE'}
        sel = strcmpi(allFmt, f{1});
        if any(sel)
            fprintf(['    %-6s : 記録%d件 (FCS検証OK=%d, ヘッダのみ推定=%d, ', ...
                     'MAC未復号(CSIのみ)=%d)\n'], ...
                f{1}, sum(sel), sum(sel & allFcsV), ...
                sum(sel & ~allFcsV & ~allNoMAC), sum(sel & allNoMAC));
        end
    end
    fprintf(['    ※あるフォーマットで FCS検証OK が 0 件の場合、そのフォーマットの\n', ...
             '      データ復号が成立していない (受信SNR不足、または変調パラメータの\n', ...
             '      推定ミス) ことを示す。\n']);
end

if stats.errors > 0
    fprintf('  エラー内訳 (上位):\n');
    ks = keys(errMsgs);
    vs = cell2mat(values(errMsgs));
    [~, sortIdx] = sort(vs, 'descend');
    for k = 1:min(5, numel(ks))
        fprintf('    %4d回: %s\n', vs(sortIdx(k)), ks{sortIdx(k)});
    end
end

%% ------------------------------------------------------------------------
%  6. 対象 SSID の BSSID を特定し、CSI を抽出
%  ------------------------------------------------------------------------
seenNetworks = struct('bssid', {}, 'ssid', {});
bssidKeys = keys(bssidToSSID);
for k = 1:numel(bssidKeys)
    seenNetworks(end+1) = struct('bssid', bssidKeys{k}, ...
        'ssid', bssidToSSID(bssidKeys{k})); %#ok<SAGROW>
end

% パケットが「その BSSID のネットワークに属する」かどうかは、送信元(Address2)
% と宛先(Address1)のどちらか一方が BSSID と一致すれば判定できる。
%   ダウンリンク (AP→クライアント): Address2(送信元) = BSSID
%   アップリンク (クライアント→AP): Address1(宛先)   = BSSID
% 送信元だけを見ると、クライアントが送るアップリンクの実データ(iperf3等)を
% 取りこぼすので注意。
allBssid = {pktLog.bssid};
allAddr1 = {pktLog.addr1};

fprintf('\n検出できたネットワーク一覧 (BSSID別の内訳):\n');
if isempty(seenNetworks)
    fprintf('  (なし)\n');
else
    isHTPkt  = strcmpi({pktLog.phyFormat}, 'HT');
    isVHTPkt = strcmpi({pktLog.phyFormat}, 'VHT');
    isHEPkt  = strcmpi({pktLog.phyFormat}, 'HE');
    for k = 1:numel(seenNetworks)
        bssid = seenNetworks(k).bssid;
        belongs = strcmp(allBssid, bssid) | strcmp(allAddr1, bssid);
        nAll  = sum(belongs);
        nHT   = sum(isHTPkt  & belongs);
        nVHT  = sum(isVHTPkt & belongs);
        nHE   = sum(isHEPkt  & belongs);
        fprintf('  BSSID=%s  SSID="%s"  (全%d件, うちHT=%d件, VHT=%d件, HE=%d件)\n', ...
            bssid, seenNetworks(k).ssid, nAll, nHT, nVHT, nHE);
    end
    fprintf(['  ※ここでの件数は MAC ヘッダまで読めた(=BSSID が判明した)パケットのみ。\n', ...
             '    複数空間ストリームのパケットは SISO 受信ではヘッダも含めて\n', ...
             '    全く復号できないため、この一覧には出てきません。\n']);
end

% --- 診断: 既知の BSSID を宛先(Address1)とする「見慣れない送信元」を列挙 ---
% これはその BSSID に接続しているクライアント端末の MAC アドレス候補
% (=アップリンク送信元) であり、Beacon を送らないため上の一覧には出てこない。
knownBssidSet = {seenNetworks.bssid};
uplinkSenders = containers.Map('KeyType', 'char', 'ValueType', 'double');
for k = 1:numel(pktLog)
    if any(strcmp(knownBssidSet, pktLog(k).addr1)) && ~any(strcmp(knownBssidSet, pktLog(k).bssid))
        uplinkSenders = bumpMap(uplinkSenders, pktLog(k).bssid);
    end
end
if uplinkSenders.Count > 0
    fprintf('  クライアント(アップリンク送信元)と思われる MAC アドレス:\n');
    ukeys = keys(uplinkSenders);
    for k = 1:numel(ukeys)
        % どの BSSID 宛てのアップリンクだったかも添える
        destBssid = '';
        for j = 1:numel(pktLog)
            if strcmp(pktLog(j).bssid, ukeys{k}) && any(strcmp(knownBssidSet, pktLog(j).addr1))
                destBssid = pktLog(j).addr1;
                break;
            end
        end
        fprintf('    %s -> BSSID=%s  (%d件)\n', ukeys{k}, destBssid, uplinkSenders(ukeys{k}));
    end
end

targetBSSID = '';
for k = 1:numel(seenNetworks)
    if strcmp(seenNetworks(k).ssid, targetSSID)
        targetBSSID = seenNetworks(k).bssid;
        break;
    end
end

% --- BSS Color と BSSID の対応を学習する ------------------------------------
% HE-SIG-A の BSS Color は CRC を通ったうえで得られる 6bit の BSS 識別子で、
% MAC ヘッダが読めなくても分かる。高MCS (1024QAM 等) では受信SNRが少し
% 足りないだけでデータ部の FCS が通らなくなるが、CSI はプリアンブル
% (HE-LTF) から算出しているので影響を受けない。そこで
%   「MAC が読めたパケット」から Color -> BSSID の対応を学習し、
%   「MAC が読めなかったパケット」を Color 一致で同じ BSS に帰属させる
% ことで、CSI を捨てずに済ませる。
%
% 6bit しかないので別々の BSS が同じ Color を使うことはあり得る。学習中に
% 1つの Color が複数の BSSID に結びついた場合は、その Color での帰属を
% 諦める (誤って別ネットワークの CSI を混ぜるより、取りこぼす方が安全)。
% 学習には次の2つの条件を満たすパケットだけを使う。
%   (a) FCS 検証まで通っている
%       高MCSではビット誤りの残った MPDU が「MACヘッダのみ」の緩い妥当性検査を
%       偶然通ることがあり、でたらめなアドレスを学習してしまう。FCS が通った
%       ものだけに限れば、この誤学習を完全に排除できる。
%   (b) 送信元(Address2)か宛先(Address1)のどちらかが、Beacon で SSID を
%       確認済みの BSSID である
%       ダウンリンクなら Address2、アップリンクなら Address1 が BSSID になる。
%       Beacon 由来の BSSID に錨を下ろすことで、クライアントの MAC アドレスを
%       誤って BSSID として登録することも防げる。
%
% 学習の根拠は2種類ある。
%   ビーコン / Probe Response … 802.11ax の AP はビーコンの HE Operation 要素で
%       自分の BSS Color を告知している。ビーコンは最も低い変調方式で送られる
%       ので、HE データが 1 件も復号できない受信品質でも確実に読める。
%       主な根拠はこちら。
%   FCS が通った HE パケット … HE-SIG-A の Color と MAC アドレスの組。
%       受信品質に余裕があるときの補助的な根拠。
allColors  = [pktLog.bssColor];
colorToBSSID   = containers.Map('KeyType', 'double', 'ValueType', 'char');
colorEvidence  = containers.Map('KeyType', 'double', 'ValueType', 'double');   % HE パケット
colorEvidenceBcn = containers.Map('KeyType', 'double', 'ValueType', 'double'); % ビーコン
colorConflicts = containers.Map('KeyType', 'double', 'ValueType', 'logical');
for k = 1:numel(pktLog)
    c = pktLog(k).bssColor;
    if c < 0 || ~pktLog(k).fcsVerified
        continue;   % Color を持たない、または MAC を確証できていないパケット
    end
    fromBeacon = ~strcmpi(pktLog(k).phyFormat, 'HE');

    % Beacon で確認済みの BSSID に一致する方を採用する
    bs = '';
    if any(strcmp(knownBssidSet, pktLog(k).bssid))
        bs = pktLog(k).bssid;
    elseif any(strcmp(knownBssidSet, pktLog(k).addr1))
        bs = pktLog(k).addr1;
    end
    if isempty(bs)
        continue;   % どちらの向きでも既知の BSS に結びつかない
    end

    if ~isKey(colorToBSSID, c)
        colorToBSSID(c)     = bs;
        colorEvidence(c)    = 0;
        colorEvidenceBcn(c) = 0;
    elseif ~strcmp(colorToBSSID(c), bs)
        % 別々の BSS が同じ Color を使っている。6bit しかないので
        % 起こり得る。この Color での帰属は諦める。
        colorConflicts(c) = true;
        continue;
    end
    if fromBeacon
        colorEvidenceBcn(c) = colorEvidenceBcn(c) + 1;
    else
        colorEvidence(c) = colorEvidence(c) + 1;
    end
end

if colorToBSSID.Count > 0
    fprintf('\nBSS Color と BSSID の対応 (ビーコンの告知と、FCS検証済みの HE パケットから学習):\n');
    cKeys = sort(cell2mat(keys(colorToBSSID)));
    for k = 1:numel(cKeys)
        bs = colorToBSSID(cKeys(k));
        ss = '';
        for j = 1:numel(seenNetworks)
            if strcmp(seenNetworks(j).bssid, bs)
                ss = seenNetworks(j).ssid;
                break;
            end
        end
        conflict = isKey(colorConflicts, cKeys(k)) && colorConflicts(cKeys(k));
        fprintf('  Color%-2d -> BSSID=%s%s  (根拠: ビーコン %d件, HEパケット %d件)%s\n', ...
            cKeys(k), bs, ...
            ternary(~isempty(ss), sprintf('  SSID="%s"', ss), ''), ...
            colorEvidenceBcn(cKeys(k)), colorEvidence(cKeys(k)), ...
            ternary(conflict, '   <-- 複数のBSSIDと衝突。Colorでの帰属は行いません', ''));
    end
elseif stats.csiOnly > 0
    fprintf(['\nBSS Color と BSSID の対応を学習できませんでした。\n', ...
             '  (ビーコンに BSS Color の告知 (HE Operation 要素) が無く、\n', ...
             '   FCS 検証まで通った HE パケットも 1 件も無いため)\n', ...
             '  MAC未復号の HE パケット %d 件は、どの SSID にも帰属させられません。\n'], ...
        stats.csiOnly);
end

matched = pktLog([]);   % 空の構造体配列
nByColor = 0;

if isempty(targetBSSID)
    warning('decodeIQ_HE:ssidNotFound', ...
        ['指定した SSID "%s" の Beacon/Probe Response を検出できませんでした。\n', ...
         'captureIQ.m の captureDuration を長くする、pktDetThreshold を下げる (例 0.3)、', ...
         'SSID の綴り・電波状況を確認する、などをお試しください。'], targetSSID);
else
    isMatch = strcmp(allBssid, targetBSSID) | strcmp(allAddr1, targetBSSID);

    % 対象 BSSID に対応する BSS Color を求め、MACが読めなかったパケットを拾う
    if useBSSColorFallback
        targetColors = [];
        cKeys = cell2mat(keys(colorToBSSID));
        for k = 1:numel(cKeys)
            if strcmp(colorToBSSID(cKeys(k)), targetBSSID) && ...
                    ~(isKey(colorConflicts, cKeys(k)) && colorConflicts(cKeys(k)))
                targetColors(end+1) = cKeys(k); %#ok<SAGROW>
            end
        end
        if ~isempty(targetColors)
            % FCS で MAC を確認できていない HE パケットは、アドレスではなく
            % Color で判定する。受信品質が足りないとき、FCS を通らなかった
            % ヘッダは壊れていることが多く、「ヘッダのみ推定」で読めた
            % アドレスも偶然の値になっているため。CSI はプリアンブル由来
            % なのでこれらも有効。
            heUnverified = strcmpi({pktLog.phyFormat}, 'HE') & ~logical([pktLog.fcsVerified]);
            byColor   = heUnverified & ismember(allColors, targetColors);
            nByColor  = sum(byColor & ~isMatch);
            isMatch   = isMatch | byColor;
        end
    end

    matched = pktLog(isMatch);
    fprintf('\n対象 SSID "%s" (BSSID=%s) のパケット数: %d (アップリンク+ダウンリンク)\n', ...
        targetSSID, targetBSSID, numel(matched));
    if nByColor > 0
        fprintf(['  うち %d 件は MAC ヘッダを FCS で確認できず、HE-SIG-A の BSS Color\n', ...
                 '  一致で帰属させたものです (CSI はプリアンブル由来なので有効。\n', ...
                 '  fcsVerified = false が目印)。\n'], nByColor);
    end
end

% --- フォーマット別に [パケット数 x サブキャリア数] の行列へまとめる ---
% Non-HT/HT は 52 本、VHT(20MHz) は 56 本、HE(20MHz) は 242 本と
% サブキャリア数が異なるため、1 つの行列には混ぜられない。
% フォーマットごとに分けて保存する。
if isempty(matched)
    allFormats   = {};
    allTimeSec   = [];
    allFrameType = {};
    allFcs       = logical([]);
else
    allFormats   = {matched.phyFormat}.';
    allTimeSec   = [matched.timeSec].';
    allFrameType = {matched.frameType}.';
    allFcs       = logical([matched.fcsVerified]).';
end

isNonHT = strcmpi(allFormats, 'Non-HT');
isHT    = strcmpi(allFormats, 'HT');
isVHT   = strcmpi(allFormats, 'VHT');
isHE    = strcmpi(allFormats, 'HE');

csiNonHT       = stackCSI(matched(isNonHT));
timeSecNonHT   = allTimeSec(isNonHT);
frameTypeNonHT = allFrameType(isNonHT);
fcsNonHT       = allFcs(isNonHT);

csiHT       = stackCSI(matched(isHT));
timeSecHT   = allTimeSec(isHT);
frameTypeHT = allFrameType(isHT);
fcsHT       = allFcs(isHT);

csiVHT       = stackCSI(matched(isVHT));
timeSecVHT   = allTimeSec(isVHT);
frameTypeVHT = allFrameType(isVHT);
fcsVHT       = allFcs(isVHT);

csiHE       = stackCSI(matched(isHE));
timeSecHE   = allTimeSec(isHE);
frameTypeHE = allFrameType(isHE);
fcsHE       = allFcs(isHE);
if isempty(matched)
    uplinkHE  = logical([]);
    bssColorHE = [];
else
    uplinkHE   = logical([matched.uplink]).';
    uplinkHE   = uplinkHE(isHE);       % true=上り(STA->AP), false=下り(AP->STA)
    bssColorHE = [matched.bssColor].';
    bssColorHE = bssColorHE(isHE);
end

if ~isempty(matched)
    fprintf('  内訳: Non-HT=%d, HT=%d, VHT=%d, HE=%d\n', ...
        size(csiNonHT, 1), size(csiHT, 1), size(csiVHT, 1), size(csiHE, 1));
    % MAC を信頼できる分だけで PPDU と MPDU の関係を見る。
    % FCS で確認できていない HE パケットは、ヘッダのみ推定で何か読めていても
    % 壊れていることが多く (帰属は BSS Color で行っている)、MPDU 数も当て
    % にならないので外す。MAC未復号のものは MPDU 数がそもそも分からない。
    heUnverifiedM = strcmpi(allFormats, 'HE') & ~allFcs;
    macKnown   = ~strcmp(allFrameType, '(MAC未復号)') & ~heUnverifiedM;
    mpduCounts = [matched.mpduCount].';
    if any(macKnown)
        fprintf(['  MACが確認できた %d 件について、PPDU(電波上の送信単位)数に対する\n', ...
                 '  集約されたMPDU(データ単位)の合計=%d\n'], ...
            sum(macKnown), sum(mpduCounts(macKnown)));
    end

    nHeaderOnly = sum(~allFcs & macKnown);
    if nHeaderOnly > 0
        fprintf(['  ※%d 件は FCS 未検証 (ペイロードにビット誤りがあり、MAC ヘッダ\n', ...
                 '    のみから送信元を判定したもの)。CSI 自体はプリアンブルから\n', ...
                 '    算出しており影響を受けません。\n'], nHeaderOnly);
    end

    % --- 上り/下りの内訳 (HE-SIG-A の UL/DL ビット) ---
    % 上りと下りでは送信している機器が違うので、電波の通り道 (= CSI) も
    % 別物になる。混ぜて時系列にすると意味が壊れるため、必ず分けて扱うこと。
    % このビットは HE-SIG-A にあり CRC で守られているので、MAC が読めなくても
    % 信頼できる。
    isHEMatched = strcmpi(allFormats, 'HE');
    if any(isHEMatched)
        upFlags = logical([matched.uplink]).';
        nUp   = sum(isHEMatched & upFlags);
        nDown = sum(isHEMatched & ~upFlags);
        fprintf('  HE の向き: 下り(AP->STA)=%d件, 上り(STA->AP)=%d件\n', nDown, nUp);
        if nUp > 0 && nDown > 0
            fprintf(['    ※上りと下りは送信機が異なるため CSI も別物です。\n', ...
                     '      解析時は保存した uplinkHE で分けてください。\n']);
        end
    end
end

% --- ResultCSI.m 互換の「主」データを決める ---
% ResultCSI.m は csi (行列) / subcarrierIndices / packetStartIndex /
% csiMeta という変数名を前提とするため、いずれか 1 形式をその名前でも保存する。
% 本スクリプトの目的は HE の CSI 収集なので、HE が 1 件でもあれば HE を
% 優先する。HE が 0 件のときだけパケット数が最多の形式にフォールバックする。
counts = [size(csiNonHT,1), size(csiHT,1), size(csiVHT,1), size(csiHE,1)];
if counts(4) > 0
    bestIdx = 4;
else
    [~, bestIdx] = max(counts);
end
formatNames = {'Non-HT', 'HT', 'VHT', 'HE'};
csiSets    = {csiNonHT, csiHT, csiVHT, csiHE};
subcSets   = {subcarrierIndicesNonHT, subcarrierIndicesHT20, ...
              subcarrierIndicesVHT20, subcarrierIndicesHE20};
timeSets   = {timeSecNonHT, timeSecHT, timeSecVHT, timeSecHE};
frameSets  = {frameTypeNonHT, frameTypeHT, frameTypeVHT, frameTypeHE};
fcsSets    = {fcsNonHT, fcsHT, fcsVHT, fcsHE};

primaryFormat     = formatNames{bestIdx};
csi               = csiSets{bestIdx};
subcarrierIndices = subcSets{bestIdx};
timeSec           = timeSets{bestIdx};
frameType         = frameSets{bestIdx};
fcsVerified       = fcsSets{bestIdx};
phyFormat         = repmat({primaryFormat}, size(csi, 1), 1);

% ResultCSI.m は packetStartIndex と sampleRate から時間軸を作る
packetStartIndex = round(timeSec(:) * sampleRate);

if ~isempty(csi)
    fprintf('  主データ(変数 csi): %s フォーマット, %d パケット x %d サブキャリア\n', ...
        primaryFormat, size(csi, 1), size(csi, 2));
end

%% ------------------------------------------------------------------------
%  7. 保存 (.mat) — HDD と USB メモリの両方へ
%  ------------------------------------------------------------------------
% 全フレームの記録 (IMU との時刻合わせ用。対象 SSID に絞る前の pktLog から作る)
frameLog = buildFrameLog(pktLog, targetBSSID);
fprintf('全フレームの記録 (frameLog): %d 件 (うち AP 宛ての上りデータ %d 件)\n', ...
    numel(frameLog.timeSec), nnz(frameLog.toAP & frameLog.isData));

% 変数名は ResultCSI.m と揃えて csiMeta とする
csiMeta = struct();
csiMeta.description      = ['CSI filtered by target SSID ', ...
    '(Non-HT / HT(NSS=1,20MHz) / SU-VHT(NSTS=1,20MHz) / HE-SU(NSTS=1,20MHz))'];
csiMeta.primaryFormat    = primaryFormat;   % 変数 csi がどの形式か
csiMeta.centerFrequency  = centerFrequency;
csiMeta.sampleRate       = sampleRate;
csiMeta.gain             = gain;
csiMeta.captureDuration  = captureDuration;
csiMeta.wifiChannel      = 36;
csiMeta.platform         = usrpPlatform;
csiMeta.serialNum        = usrpSerialNum;
csiMeta.pktDetThreshold  = pktDetThreshold;
csiMeta.overrunCount     = overrunCount;
csiMeta.decodeStats      = stats;
csiMeta.heSupported      = heAvailable;       % HE 復号が有効だったか
csiMeta.useBSSColorFallback = useBSSColorFallback;
csiMeta.numByBSSColor    = nByColor;          % BSS Color 一致で帰属させた件数
csiMeta.bssColorMap      = colorToBSSID;      % 学習した Color -> BSSID
csiMeta.captureDatetime  = timestamp;         % キャプチャ時刻
csiMeta.decodeDatetime   = datestr(now, 'yyyymmddHHMM');   % 復号を行った時刻
csiMeta.sourceRawFile    = inputRawFile;      % どの生IQから作られたか
csiMeta.captureMeta      = meta;              % キャプチャ時の全メタデータ
csiMeta.decodedBy        = 'decodeIQ_HE.m';
csiMeta.matlabVersion    = version;

targetSSIDOut  = targetSSID;  %#ok<NASGU>
targetBSSIDOut = targetBSSID; %#ok<NASGU>

saveVars = { ...
    'csi', 'subcarrierIndices', 'packetStartIndex', 'csiMeta', ...
    'phyFormat', 'fcsVerified', 'timeSec', 'frameType', ...
    'csiNonHT', 'timeSecNonHT', 'frameTypeNonHT', 'fcsNonHT', ...
    'csiHT', 'timeSecHT', 'frameTypeHT', 'fcsHT', ...
    'csiVHT', 'timeSecVHT', 'frameTypeVHT', 'fcsVHT', ...
    'csiHE', 'timeSecHE', 'frameTypeHE', 'fcsHE', 'uplinkHE', 'bssColorHE', ...
    'subcarrierIndicesNonHT', 'subcarrierIndicesHT20', ...
    'subcarrierIndicesVHT20', 'subcarrierIndicesHE20', ...
    'targetSSIDOut', 'targetBSSIDOut', 'seenNetworks', 'frameLog'};

fprintf('\n');
nSaved = 0;
for k = 1:numel(outMatFiles)
    try
        save(outMatFiles{k}, saveVars{:}, '-v7.3');
        fprintf('結果を保存しました: %s\n', outMatFiles{k});
        nSaved = nSaved + 1;
    catch ME
        % 片方 (USB の抜き差し等) が失敗しても、もう片方は残す
        warning('decodeIQ_HE:saveFailed', ...
            '保存に失敗しました (%s): %s', outMatFiles{k}, ME.message);
    end
end

if nSaved == 0
    error('decodeIQ_HE:allSavesFailed', ...
        'すべての出力先への保存に失敗しました。ドライブの接続を確認してください。');
end

fprintf('すべての処理が完了しました。\n');
if ~isempty(decodeLog.file)
    fprintf('この画面の内容は次のファイルにも保存しました:\n  %s\n', decodeLog.file);
end
stopDecodeLog(decodeLog);

%% ------------------------------------------------------------------------
%  ローカル関数
%  ------------------------------------------------------------------------
function logState = startDecodeLog(logFile)
    % 画面に表示する内容を logFile にも記録し始める (MATLAB の diary)。
    %
    % diary は同時に 1 つのファイルにしか書けない。一括処理
    % (decodeIQ_HE_batch.m) が自分のログを取っている最中に呼ばれることが
    % あるので、そのとき動いていた diary を控えておき、stopDecodeLog で戻す。
    logState = struct('file', '', ...
        'prevFile', get(0, 'DiaryFile'), ...
        'prevOn',   strcmp(get(0, 'Diary'), 'on'));
    try
        folder = fileparts(logFile);
        if ~isempty(folder) && ~exist(folder, 'dir')
            mkdir(folder);
        end
        diary off;
        if isfile(logFile)
            delete(logFile);   % diary は追記するので、やり直しのときは消しておく
        end
        diary(logFile);
        logState.file = logFile;
    catch ME
        warning('decodeIQ_HE:logFailed', ...
            'ログファイルを作れませんでした (%s)。画面表示のみで続けます: %s', ...
            logFile, ME.message);
        if logState.prevOn
            diary(logState.prevFile);
        end
    end
end

function stopDecodeLog(logState)
    % startDecodeLog で始めた記録を閉じ、その前に動いていた diary に戻す
    if isempty(logState.file)
        return;
    end
    diary off;
    if logState.prevOn && ~isempty(logState.prevFile)
        diary(logState.prevFile);
    end
end

function src = openIQSource(input, meta)
    % 生IQの読み出し口を作る。
    %   input が数値       … *_raw.mat から読んだ complex double (全部メモリ上)
    %   input がファイルパス … captureIQ_single.m の *_raw.bin (必要な分だけ読む)
    % どちらも readIQ(src, 開始サンプル, サンプル数) で同じように読める。
    src = struct('isBin', false, 'data', [], 'fid', -1, ...
        'precision', 'int16', 'scale', 1, 'bytesPerSample', 16, 'totalSamples', 0);

    if isnumeric(input)
        src.data         = input(:);
        src.totalSamples = numel(src.data);
        return;
    end

    src.isBin         = true;
    src.precision     = char(string(getProp(meta, 'rawPrecision', 'int16')));
    src.scale         = double(getProp(meta, 'rawScaleFactor', 1/32768));
    src.bytesPerSample = double(getProp(meta, 'bytesPerSample', 4));

    src.fid = fopen(input, 'r');
    if src.fid == -1
        error('decodeIQ_HE:cannotOpenBin', ...
            '生データ (.bin) を開けませんでした: %s', input);
    end

    % 実ファイルサイズとメタの記録値の小さい方を採用する。
    % (キャプチャが途中で止まった場合にメタの値が過大になり得るため)
    d = dir(input);
    nFromFile = floor(d.bytes / src.bytesPerSample);
    nFromMeta = double(getProp(meta, 'totalSamples', nFromFile));
    src.totalSamples = min(nFromFile, max(nFromMeta, 0));
end

function v = readIQ(src, startSample, numSamples)
    % startSample (0-based) から numSamples 個を complex double で返す。
    % 範囲外は自動で切り詰める。
    numSamples = min(numSamples, src.totalSamples - startSample);
    if numSamples <= 0
        v = complex(zeros(0, 1));
        return;
    end

    if ~src.isBin
        v = src.data(startSample + (1:numSamples));
        return;
    end

    % .bin は I,Q が交互に並んだ整数列。読みながら double 化してスケールを戻す。
    fseek(src.fid, startSample * src.bytesPerSample, 'bof');
    raw = fread(src.fid, 2 * numSamples, [src.precision '=>double']);
    n   = floor(numel(raw) / 2);
    if n == 0
        v = complex(zeros(0, 1));
        return;
    end
    v = complex(raw(1:2:2*n), raw(2:2:2*n)) * src.scale;
end

function closeIQSource(src)
    if src.isBin && src.fid ~= -1
        try
            fclose(src.fid);
        catch
        end
    end
end

function M = stackCSI(entries)
    % パケットごとの CSI 行ベクトルを [パケット数 x サブキャリア数] の行列に積む。
    % サブキャリア数が揃わないものが混ざっていた場合は最頻の長さに合わせる。
    if isempty(entries)
        M = [];
        return;
    end
    lens = cellfun(@numel, {entries.csi});
    n = mode(lens);
    keep = (lens == n);
    M = cat(1, entries(keep).csi);
end

function m = bumpMap(m, key)
    % containers.Map のカウンタを 1 増やす (キーが無ければ 1 で作る)
    if isKey(m, key)
        m(key) = m(key) + 1;
    else
        m(key) = 1;
    end
end

function printCountMapNum(m, label, fmt)
    % 数値キーの集計 Map をキー昇順で 1 行に表示する
    if m.Count == 0
        return;
    end
    kk = sort(cell2mat(keys(m)));
    fprintf('%s', label);
    for k = 1:numel(kk)
        fprintf(fmt, kk(k), m(kk(k)));
    end
    fprintf('\n');
end

function printCountMapChar(m, label)
    % 文字列キーの集計 Map を 1 行に表示する
    if m.Count == 0
        return;
    end
    kk = keys(m);
    fprintf('%s', label);
    for k = 1:numel(kk)
        fprintf('%s=%d回  ', kk{k}, m(kk{k}));
    end
    fprintf('\n');
end

function [cnt, shown] = recordDeag(cnt, shown, fmtName, info)
    % A-MPDU 分解の結果を集計する。ここが失敗しているか成功しているかで、
    % 「データ復号自体が壊れている」のか「FCS だけ通らない」のかを切り分ける。
    % deagStatus が空 = 分解まで到達せず (SIG段階で棄却) なので集計しない
    if ~isempty(info.deagStatus)
        cnt = bumpMap(cnt, sprintf('%s: %s', fmtName, info.deagStatus));
    end

    % 想定外のデータ形式は一度だけ警告する (原因究明の手掛かりになるため)。
    % ビット列と16進文字列はどちらも扱えるので、ここに来るのはそれ以外。
    if info.nonBinary && ~shown
        shown = true;
        fprintf(['    <警告> 分解した MPDU がビット列でも16進文字列でもありません。\n', ...
                 '            class=%s, 値域=[%g %g]\n', ...
                 '            この形式は MAC ヘッダとして解釈できないため除外します。\n'], ...
            info.dataClass, info.dataRange(1), info.dataRange(2));
    end
end

function y = applyCFO(x, fs, cfoHz)
    % x に周波数オフセット cfoHz [Hz] 分の位相回転を与える (補正には -cfo を渡す)
    n = (0:numel(x)-1).';
    y = x .* exp(1j * 2 * pi * cfoHz * n / fs);
end

function color = parseBeaconBSSColor(psduBits)
    % ビーコン / Probe Response の本体から HE Operation 要素を探し、
    % AP が告知している BSS Color (0〜63) を返す。見つからなければ -1。
    %
    % HE Operation 要素 (IEEE 802.11ax, Element ID 255 / Extension ID 36):
    %   [ID=255][長さ][拡張ID=36][HE Operation Parameters 3byte][BSS Color Information 1byte]...
    %   BSS Color Information: bit0-5 = BSS Color, bit6 = Partial, bit7 = Disabled
    % FCS は呼び出し前に wlanMPDUDecode で確認済みなので、中身は信頼できる。
    color = -1;
    [dfmt, oct] = mpduDataFormat(psduBits);
    if isempty(dfmt)
        return;
    end
    oct = double(oct(:)).';

    hdrLen = 24;
    if numel(oct) >= 2 && bitand(oct(2), 128) ~= 0
        hdrLen = 28;   % Order ビットが立っていれば HT Control が付く
    end
    % 本体の先頭は固定フィールド: Timestamp 8 + Beacon Interval 2 + Capability 2
    p    = hdrLen + 12 + 1;
    last = numel(oct) - 4;   % 末尾 4 バイトは FCS
    while p + 1 <= last
        id  = oct(p);
        len = oct(p + 1);
        if p + 1 + len > last
            return;   % 要素の長さが本体からはみ出している
        end
        if id == 255 && len >= 5 && oct(p + 2) == 36
            info = oct(p + 6);   % 拡張ID(1) + HE Operation Parameters(3) の次
            if bitand(info, 128) == 0   % BSS Color Disabled でない
                color = bitand(info, 63);
            end
            return;
        end
        p = p + 2 + len;
    end
end

function [status, consumed, entry] = processNonHT(pkt, chanEst, noiseEst, chanBW)
    % Non-HT (802.11a/g) パケットを復号する。
    %   status: 1=成功(entryあり), 0=失敗/読み捨て, -1=サンプル不足
    %   consumed: pkt 先頭から消費したサンプル数
    entry = [];
    consumed = 400;   % L-STF..L-SIG

    % --- L-SIG 復号 (レート・PSDU長を取得) ---
    [recLSIGBits, lsigFail] = wlanLSIGRecover(pkt(321:400), chanEst, noiseEst, chanBW);
    if lsigFail
        status = 0;
        return;
    end

    [mcs, psduLen] = decodeLSIGBits(recLSIGBits);
    if isempty(mcs) || psduLen <= 0 || psduLen > 4095
        status = 0;
        return;
    end

    try
        cfgNonHT = wlanNonHTConfig('MCS', mcs, 'PSDULength', psduLen, ...
            'ChannelBandwidth', chanBW);
        ind = wlanFieldIndices(cfgNonHT);
    catch
        status = 0;
        return;
    end

    if numel(pkt) < ind.NonHTData(2)
        status = -1;
        return;   % データ部がキャプチャ範囲外
    end
    consumed = double(ind.NonHTData(2));

    rxPSDU = wlanNonHTDataRecover(pkt(ind.NonHTData(1):ind.NonHTData(2)), ...
        chanEst, noiseEst, cfgNonHT);

    % 未対応サブタイプの警告は頻出するためここでは抑止する (サマリで集計)
    wState = warning('off', 'all');
    [cfgMAC, payload, mpduStatus] = wlanMPDUDecode(rxPSDU, cfgNonHT, 'DataFormat', 'bits');
    warning(wState);

    if ~strcmpi(string(mpduStatus), "Success")
        status = 0;
        return;
    end

    entry = buildEntry(cfgMAC, payload, 'Non-HT', chanEst);

    % ビーコン / Probe Response なら、AP が告知している BSS Color を読む。
    % HE の CSI を対象 SSID に帰属させるときの、確実な手掛かりになる。
    if ~isempty(entry) && any(strcmpi(entry.frameType, {'Beacon', 'Probe Response'}))
        entry.bssColor = parseBeaconBSSColor(rxPSDU);
    end
    status = 1;
end

function [status, consumed, entry, info] = processHT(pkt, lltfChanEst, noiseEst, chanBW)
    % HT-Mixed (802.11n, NSS=1, 20MHz) パケットを復号する。
    %   status: 1=成功, 0=失敗/非対応, -1=サンプル不足
    %   info:   診断用 (棄却理由・観測されたMCS/NSS/帯域幅)
    entry = [];
    consumed = 560;   % L-STF..HT-SIG
    info = struct('reason', '', 'mcs', -1, 'nss', 0, 'cbw40', false, ...
        'coding', 0, 'gi', 0, 'aggregated', false, 'reservedOK', false, ...
        'mpduCount', 0, 'deagStatus', '', ...
        'nonBinary', false, 'dataClass', '', 'dataRange', [0 0]);

    % --- HT-SIG 復号 (2シンボル, pkt(401:560)) ---
    [htsigBits, htsigFail] = wlanHTSIGRecover(pkt(401:560), lltfChanEst, noiseEst, chanBW);
    if htsigFail
        info.reason = 'HT-SIG CRC失敗';
        status = 0;
        return;
    end

    ht = parseHTSIGBits(htsigBits);
    info.mcs        = ht.mcs;
    info.nss        = ht.nss;
    info.cbw40      = ht.cbw40;
    info.coding     = ht.coding;
    info.gi         = ht.gi;
    info.aggregated = ht.aggregated;
    info.reservedOK = ht.reservedOK;

    % 以下はいずれも SISO(受信1本)/20MHz 受信では扱えない構成
    if ~ht.isValid
        info.reason = 'HT-SIG解釈不能';
        status = 0; return;
    elseif ht.cbw40
        info.reason = '帯域幅40MHz(20MHz受信では復号不可)';
        status = 0; return;
    elseif ht.nss ~= 1
        info.reason = sprintf('NSS=%d(アンテナ1本では復号不可)', ht.nss);
        status = 0; return;
    elseif ht.psduLen <= 0 || ht.psduLen > 65535
        info.reason = 'HT-LENGTH不正';
        status = 0; return;
    end

    codingStr = ternary(ht.coding == 1, 'LDPC', 'BCC');
    giStr     = ternary(ht.gi == 1, 'Short', 'Long');

    try
        cfgHT = wlanHTConfig('ChannelBandwidth', chanBW, ...
            'NumTransmitAntennas', 1, 'NumSpaceTimeStreams', 1, ...
            'MCS', ht.mcs, 'PSDULength', ht.psduLen, ...
            'ChannelCoding', codingStr, 'GuardInterval', giStr);
        ind = wlanFieldIndices(cfgHT);
    catch
        info.reason = 'HTコンフィグ生成失敗';
        status = 0;
        return;
    end

    if numel(pkt) < ind.HTData(2)
        status = -1;
        return;   % データ部がまだ届いていない -> 次フレーム待ち
    end
    consumed = double(ind.HTData(2));

    % --- HT-LTF を復調してチャネル推定 (= CSI) ---
    htDemod   = wlanHTLTFDemodulate(pkt(ind.HTLTF(1):ind.HTLTF(2)), cfgHT);
    htChanEst = wlanHTLTFChannelEstimate(htDemod, cfgHT);

    rxPSDU = wlanHTDataRecover(pkt(ind.HTData(1):ind.HTData(2)), htChanEst, noiseEst, cfgHT);

    % --- HT の PSDU は A-MPDU の場合があるので分解してから MPDU を復号 ---
    [cfgMAC, payload, ok, deagInfo] = decodeAggregatedPSDU(rxPSDU, cfgHT);
    info.mpduCount  = deagInfo.mpduCount;
    info.deagStatus = deagInfo.status;
    info.nonBinary  = deagInfo.nonBinary;
    info.dataClass  = deagInfo.dataClass;
    info.dataRange  = deagInfo.dataRange;
    if ~ok
        info.reason = sprintf('MPDU復号失敗(分解=%s,MPDU数=%d)', ...
            deagInfo.status, deagInfo.mpduCount);
        status = 0;
        return;
    end

    entry = buildEntry(cfgMAC, payload, 'HT', htChanEst);
    if ~isempty(entry)
        entry.fcsVerified = ~deagInfo.headerOnly;
        entry.mpduCount   = deagInfo.mpduCount;   % 0 = A-MPDU分解失敗
    end
    status = 1;
end

function out = parseHTSIGBits(bits)
    % HT-SIG (48bit = HT-SIG1(24) + HT-SIG2(24)) を解釈する。
    % IEEE Std 802.11-2016, 19.3.9.4 (HT-SIG) に基づく。多ビットフィールドは
    % LSB 先頭で送信される。
    %   bits(1:7)   MCS (0-31。0-7=1ストリーム,8-15=2,16-23=3,24-31=4)
    %   bits(8)     CBW (0=20MHz, 1=40MHz)
    %   bits(9:24)  HT-LENGTH (PSDU長 [byte] を直接表す)
    %   bits(28)    Aggregation (1=HT-DataがA-MPDU)
    %   bits(31)    FEC Coding (0=BCC, 1=LDPC)
    %   bits(32)    Short GI
    bits = double(bits(:)).';
    out = struct('isValid', false, 'mcs', -1, 'nss', 0, 'cbw40', false, ...
        'psduLen', 0, 'coding', 0, 'gi', 0, 'aggregated', false, ...
        'reservedOK', false);

    if numel(bits) < 34
        return;
    end

    % 自己チェック: HT-SIG2 の Reserved ビット (全体で27ビット目) は仕様上 1 固定
    out.reservedOK = bits(27) == 1;

    mcsBits = bits(1:7);
    out.mcs   = sum(mcsBits .* 2.^(0:6));
    out.nss   = 1 + floor(out.mcs / 8);
    out.cbw40 = bits(8) == 1;

    lenBits = bits(9:24);   % LSB先頭
    out.psduLen = sum(lenBits .* 2.^(0:15));

    out.aggregated = bits(28) == 1;
    out.coding      = bits(31);
    out.gi           = bits(32);
    out.isValid = true;
end

function [status, consumed, entry, info] = processVHT(pkt, lltfChanEst, noiseEst, chanBW)
    % SU-VHT (802.11ac, NSTS=1, 20MHz) パケットを復号する。
    %   status: 1=成功, 0=失敗/非対応, -1=サンプル不足
    %   info:   診断用 (棄却理由・観測された帯域幅/NSTS/MCS)
    entry = [];
    consumed = 560;   % L-STF..VHT-SIG-A
    info = struct('reason', '', 'bwMHz', 0, 'nsts', 0, 'mcs', 0, ...
        'reservedOK', false, 'gi', 0, 'coding', 0, 'lsigLen', 0, 'nsym', 0, ...
        'apepFromLSIG', 0, 'apepFromSIGB', 0, 'apepUsed', 0, ...
        'mpduCount', 0, 'deagStatus', '', ...
        'nonBinary', false, 'dataClass', '', 'dataRange', [0 0]);

    % --- VHT-SIG-A 復号 (2シンボル) ---
    [sigaBits, sigaFail] = wlanVHTSIGARecover(pkt(401:560), lltfChanEst, noiseEst, chanBW);
    if sigaFail
        info.reason = 'SIG-A CRC失敗';
        status = 0;
        return;
    end

    vhtA = parseVHTSIGABits(sigaBits);
    info.bwMHz      = vhtA.bwMHz;
    info.nsts       = vhtA.nsts;
    info.mcs        = vhtA.mcs;
    info.reservedOK = vhtA.reservedOK;
    info.gi         = vhtA.gi;
    info.coding     = vhtA.coding;

    % 以下はいずれも SISO(受信1本)/20MHz 受信では扱えない構成
    if ~vhtA.isValid
        info.reason = 'SIG-A解釈不能';
        status = 0; return;
    elseif ~vhtA.is20MHz
        info.reason = sprintf('帯域幅%dMHz(20MHz受信では復号不可)', vhtA.bwMHz);
        status = 0; return;
    elseif ~vhtA.isSU
        info.reason = 'MU-MIMO';
        status = 0; return;
    elseif vhtA.nsts ~= 1
        info.reason = sprintf('NSTS=%d(アンテナ1本では復号不可)', vhtA.nsts);
        status = 0; return;
    elseif vhtA.stbc
        info.reason = 'STBC';
        status = 0; return;
    end

    codingStr = ternary(vhtA.coding == 1, 'LDPC', 'BCC');
    giStr     = ternary(vhtA.gi == 1, 'Short', 'Long');

    % --- VHT-STF/LTF/SIG-B の位置を取得 ---
    %     これらの位置は帯域幅と NSTS だけで決まり APEPLength に依存しないため、
    %     暫定の APEPLength を持つコンフィグから求めてよい。
    try
        cfgProbe = wlanVHTConfig('ChannelBandwidth', chanBW, ...
            'NumTransmitAntennas', 1, 'NumSpaceTimeStreams', 1, ...
            'MCS', vhtA.mcs, 'APEPLength', 1024, ...
            'ChannelCoding', codingStr, 'GuardInterval', giStr);
        indProbe = wlanFieldIndices(cfgProbe);
    catch
        info.reason = 'VHTコンフィグ生成失敗';
        status = 0;
        return;
    end

    if numel(pkt) < indProbe.VHTSIGB(2)
        status = -1;
        return;
    end

    % --- VHT-LTF を復調してチャネル推定 (= CSI) ---
    demodVHTLTF = wlanVHTLTFDemodulate(pkt(indProbe.VHTLTF(1):indProbe.VHTLTF(2)), chanBW, 1);
    vhtChanEst  = wlanVHTLTFChannelEstimate(demodVHTLTF, chanBW, 1);

    % --- パケット長の決定 ---
    % MathWorks の helperVHTConfigRecover と同様、L-SIG と VHT-SIG-A から
    % 求めるのを主とする。VHT-SIG-B の Length は参考値として併記する。
    [lsigBits, lsigFail] = wlanLSIGRecover(pkt(321:400), lltfChanEst, noiseEst, chanBW);
    if ~lsigFail
        [~, info.lsigLen] = decodeLSIGBits(lsigBits);
    end

    % VHT PPDU の受信時間 [us] (IEEE 802.11ac の L-SIG LENGTH との関係)
    %   RXTIME = 4*ceil((L_LENGTH+3)/3) + 20
    ndbps  = vhtNDBPS20MHz(vhtA.mcs);
    if info.lsigLen > 0 && ~isempty(ndbps)
        % RXTIME [us] からプリアンブル (L-STF..VHT-SIG-B) を引いてデータ部の
        % 継続時間を求める。20MHz/NSTS=1 のプリアンブルは 40us。
        rxTimeUs   = 4 * ceil((info.lsigLen + 3) / 3) + 20;
        dataTimeUs = rxTimeUs - indProbe.VHTSIGB(2) / 20;   % 20 samples/us

        if vhtA.gi == 1
            % Short GI (3.6us/シンボル)。データ部は 4us 境界までパディング
            % されるため、パディングが 3.6us 以上になる NSYM ≡ 9 (mod 10) の
            % ときだけ floor が 1 多くなる。この曖昧性を VHT-SIG-A2 B1
            % (Short GI NSYM Disambiguation) が示すので、それで補正する。
            info.nsym = floor(dataTimeUs / 3.6);
            if vhtA.sgiDisamb == 1
                info.nsym = info.nsym - 1;
            end
        else
            info.nsym = round(dataTimeUs / 4);
        end

        if info.nsym > 0
            if vhtA.coding == 1
                % LDPC: NSYM = ceil((8*APEP + 16)/NDBPS) + (追加シンボル)
                %       末尾ビット(6bit)が無く、LDPC Extra OFDM Symbol
                %       (VHT-SIG-A2 B3) の分は APEP に寄与しない。
                nsymData = info.nsym - vhtA.ldpcExtra;
                if nsymData > 0
                    info.apepFromLSIG = floor((nsymData * ndbps - 16) / 8);
                end
            else
                % BCC: NSYM = ceil((8*APEP + 16 + 6*NES)/NDBPS), 20MHz は NES=1
                info.apepFromLSIG = floor((info.nsym * ndbps - 22) / 8);
            end
        end
    end

    % --- VHT-SIG-B 復号 (参考・比較用) ---
    try
        sigbBits = wlanVHTSIGBRecover(pkt(indProbe.VHTSIGB(1):indProbe.VHTSIGB(2)), ...
            vhtChanEst, noiseEst, chanBW);
        sigbLen = decodeVHTSIGBLength(sigbBits);
        if ~isempty(sigbLen)
            info.apepFromSIGB = sigbLen;
        end
    catch
    end

    % L-SIG 由来を優先し、駄目なら VHT-SIG-B 由来を使う
    apepCandidates = [info.apepFromLSIG, info.apepFromSIGB];
    apepCandidates = apepCandidates(apepCandidates > 0);
    if isempty(apepCandidates)
        info.reason = 'パケット長を決定できず';
        status = 0;
        return;
    end

    cfgVHT = [];
    for c = 1:numel(apepCandidates)
        try
            cfgTry = wlanVHTConfig('ChannelBandwidth', chanBW, ...
                'NumTransmitAntennas', 1, 'NumSpaceTimeStreams', 1, ...
                'MCS', vhtA.mcs, 'APEPLength', apepCandidates(c), ...
                'ChannelCoding', codingStr, 'GuardInterval', giStr);
            indTry = wlanFieldIndices(cfgTry);
            if numel(pkt) >= indTry.VHTData(2)
                cfgVHT = cfgTry;
                ind = indTry;
                info.apepUsed = apepCandidates(c);
                break;
            end
        catch
            % この候補は不正。次の候補へ
        end
    end

    if isempty(cfgVHT)
        info.reason = sprintf('APEPLength不正/長さ不足(LSIG=%d,SIGB=%d)', ...
            info.apepFromLSIG, info.apepFromSIGB);
        status = -1;   % データがまだ揃っていない可能性もある
        return;
    end

    consumed = double(ind.VHTData(2));

    rxPSDU = wlanVHTDataRecover(pkt(ind.VHTData(1):ind.VHTData(2)), ...
        vhtChanEst, noiseEst, cfgVHT);

    % --- VHT の PSDU は A-MPDU なので分解してから MPDU を復号 ---
    [cfgMAC, payload, ok, deagInfo] = decodeAggregatedPSDU(rxPSDU, cfgVHT);
    info.mpduCount  = deagInfo.mpduCount;
    info.deagStatus = deagInfo.status;
    info.nonBinary  = deagInfo.nonBinary;
    info.dataClass  = deagInfo.dataClass;
    info.dataRange  = deagInfo.dataRange;
    if ~ok
        info.reason = sprintf('MPDU復号失敗(分解=%s,MPDU数=%d)', ...
            deagInfo.status, deagInfo.mpduCount);
        status = 0;
        return;
    end

    entry = buildEntry(cfgMAC, payload, 'VHT', vhtChanEst);
    if ~isempty(entry)
        entry.fcsVerified = ~deagInfo.headerOnly;
        entry.mpduCount   = deagInfo.mpduCount;   % 0 = A-MPDU分解失敗
    end
    status = 1;
end


function [status, consumed, entry, info] = processHE(pkt, lltfChanEst, noiseEst, chanBW, fmtStr)
    % HE-SU / HE-EXT-SU (802.11ax, NSTS=1, 20MHz) パケットを復号する。
    %   status: 1=成功, 0=失敗/非対応, -1=サンプル不足
    %   info:   診断用 (棄却理由・観測されたMCS/NSTS/帯域幅など)
    %
    % 手順は MathWorks の "Recovery Procedure for an 802.11ax Packet" と同じ。
    % VHT と違い SIG フィールドのビット解釈は WLAN Toolbox
    % (interpretHESIGABits) に任せられるので、自前パーサは不要。
    entry = [];
    consumed = 640;   % L-STF..HE-SIG-A(2シンボル)
    info = struct('reason', '', 'packetFormat', char(fmtStr), 'sigaParsed', false, ...
        'bwMHz', 0, 'nsts', 0, 'mcs', -1, 'coding', '', 'giUs', 0, ...
        'heltfType', 0, 'dcm', false, 'stbc', false, 'bssColor', -1, ...
        'uplink', false, 'lsigLen', 0, 'psduLen', 0, 'numSubcarriers', 0, ...
        'pilotTracked', false, 'mpduStatus', {{}}, 'hdrReason', {{}}, 'mpduBytes', [], ...
        'subcIdx', [], 'mpduCount', 0, 'deagStatus', '', ...
        'nonBinary', false, 'dataClass', '', 'dataRange', [0 0]);

    % --- 本スクリプトで扱う HE PPDU 形式かを先に判定する ---
    % HE の PPDU 形式でプリアンブル長 (HE-SIG-A のシンボル数) が変わる。
    %   HE-SU / HE-MU / HE-TB … HE-SIG-A は 2 シンボル
    %   HE-EXT-SU (ER SU)     … HE-SIG-A は 2 シンボルを 2 回繰り返して 4 シンボル
    switch upper(char(fmtStr))
        case 'HE-SU'
            sigaSyms = 2;
        case 'HE-EXT-SU'
            sigaSyms = 4;
        case 'HE-MU'
            % HE-SIG-B から RU 割り当てとユーザ情報を読む必要があり、
            % SISO・単一ユーザ前提の本スクリプトの対象外。
            info.reason = 'HE-MU(OFDMA/MU-MIMO)';
            status = 0; return;
        case 'HE-TB'
            % 送信パラメータは AP が送った Trigger フレーム側にあるため、
            % HE-TB PPDU 単体からは復号条件が決まらない。
            info.reason = 'HE-TB(トリガベース。Triggerフレーム無しでは復号不可)';
            status = 0; return;
        otherwise
            info.reason = sprintf('未対応のHE形式(%s)', char(fmtStr));
            status = 0; return;
    end

    consumed = 480 + sigaSyms * 80;
    if numel(pkt) < consumed
        status = -1;
        return;
    end

    % --- L-SIG + RL-SIG: pre-HE のチャネル推定と L-SIG LENGTH の取得 ---
    % HE PPDU の L-SIG / RL-SIG / HE-SIG-A は 20MHz あたり 56 本の
    % サブキャリア (レガシーの 52 本 + 両端に 4 本) で送られる。
    % L-LTF の推定値は 52 本しか無いので、追加の 4 本を含む推定値を作る。
    % 復号したパラメータを入れていく器。PPDU形式と帯域幅だけ先に確定させ、
    % 残りは L-SIG と HE-SIG-A の中身で順に埋めていく。
    cfgRx = wlanHERecoveryConfig('PacketFormat', char(fmtStr), ...
        'ChannelBandwidth', chanBW);

    lsigDemod    = wlanHEDemodulate(pkt(321:480), 'L-SIG', chanBW);   % L-SIG + RL-SIG
    preInfo      = wlanHEOFDMInfo('L-SIG', chanBW);
    chanEstPreHE = preHEChannelEstimateCompat(lsigDemod, lltfChanEst, chanBW);
    % L-SIG にはパイロットによる位相追跡をかけない。
    %   - L-LTF の直後 (8us 後) にあるので、残留 CFO による位相の回りは無視できる
    %   - wlanHETrackPilotError は L-SIG に対してレガシーの 52 本のチャネル推定を
    %     要求するが、ここにあるのは HE 用の 56 本の推定なので形式が合わない
    % 位相追跡が効くのは、数 ms 続く HE-Data (と HE-SIG-A) の方。

    [eqLSIG, csiLSIG] = heEqualizeCompat(lsigDemod(preInfo.DataIndices, :, :), ...
        chanEstPreHE(preInfo.DataIndices, :), noiseEst, chanBW, 'L-SIG');
    [lsigFail, lsigInfo] = lsigBitRecoverCompat(eqLSIG, noiseEst, csiLSIG);
    if lsigFail
        info.reason = 'L-SIG パリティ失敗';
        status = 0;
        return;
    end
    info.lsigLen = double(lsigInfo.Length);

    % --- HE-SIG-A の復号 ---
    % HE-SIG-A は pre-HE のサブキャリアに載るので、チャネル推定は
    % L-SIG と同じ chanEstPreHE を使う。
    sigaDemod = wlanHEDemodulate(pkt(481 : 480 + sigaSyms * 80), 'HE-SIG-A', chanBW);
    sigaInfo  = wlanHEOFDMInfo('HE-SIG-A', chanBW);
    sigaDemod = heTrackPilotErrorCompat(sigaDemod, chanEstPreHE, cfgRx, 'HE-SIG-A');
    [eqSIGA, csiSIGA] = heEqualizeCompat(sigaDemod(sigaInfo.DataIndices, :, :), ...
        chanEstPreHE(sigaInfo.DataIndices, :), noiseEst, chanBW, 'HE-SIG-A');

    [sigaBits, sigaFail] = heSIGABitRecoverCompat(eqSIGA, noiseEst, csiSIGA);
    if sigaFail
        info.reason = 'HE-SIG-A CRC失敗';
        status = 0;
        return;
    end

    % --- HE-SIG-A のビット解釈 (WLAN Toolbox に任せる) ---
    % L-SIG LENGTH はデータ部のシンボル数の算出に必要なので、解釈前に渡す。
    cfgRx.LSIGLength = info.lsigLen;

    [cfgRx, failInterp] = interpretHESIGABits(cfgRx, sigaBits);
    if failInterp
        info.reason = 'HE-SIG-A解釈不能';
        status = 0;
        return;
    end
    info.sigaParsed = true;

    % --- 解釈できたパラメータを診断用に控える ---
    info.bwMHz     = bwStringToMHz(getProp(cfgRx, 'ChannelBandwidth', chanBW));
    info.nsts      = double(getProp(cfgRx, 'NumSpaceTimeStreams', 0));
    info.mcs       = double(getProp(cfgRx, 'MCS', -1));
    info.coding    = char(string(getProp(cfgRx, 'ChannelCoding', '')));
    info.giUs      = double(getProp(cfgRx, 'GuardInterval', 0));
    info.heltfType = double(getProp(cfgRx, 'HELTFType', 0));
    info.dcm       = logical(getProp(cfgRx, 'DCM', false));
    info.stbc      = logical(getProp(cfgRx, 'STBC', false));
    info.bssColor  = double(getProp(cfgRx, 'BSSColor', -1));
    info.uplink    = logical(getProp(cfgRx, 'UplinkIndication', false));

    % --- SISO(受信1本)/20MHz 受信では扱えない構成をここで弾く ---
    if info.bwMHz ~= bwStringToMHz(chanBW)
        info.reason = sprintf('帯域幅%dMHz(20MHz受信では復号不可)', info.bwMHz);
        status = 0; return;
    elseif info.nsts ~= 1
        info.reason = sprintf('NSTS=%d(アンテナ1本では復号不可)', info.nsts);
        status = 0; return;
    elseif info.stbc
        info.reason = 'STBC';
        status = 0; return;
    end

    % --- フィールド位置と PSDU 長 ---
    try
        ind = wlanFieldIndices(cfgRx);
    catch ME
        info.reason = ['フィールド位置算出失敗: ' ME.message];
        status = 0; return;
    end

    try
        info.psduLen = double(getPSDULength(cfgRx));
    catch
        info.psduLen = 0;
    end
    if info.psduLen <= 0
        info.reason = 'PSDU長を決定できず';
        status = 0; return;
    end

    if numel(pkt) < ind.HEData(2)
        status = -1;
        return;   % データ部がまだ届いていない (末尾で切れている)
    end

    % パケット末尾は HE-Data の後ろに PE (Packet Extension) が付く。
    % 次のパケット探索を PE の後ろから始めるため、あれば末尾に含める。
    lastSample = double(ind.HEData(2));
    if isfield(ind, 'HEPE') && ~isempty(ind.HEPE) && numel(pkt) >= double(ind.HEPE(2))
        lastSample = max(lastSample, double(ind.HEPE(2)));
    end
    consumed = lastSample;

    % --- HE-LTF を復調してチャネル推定 (= CSI) ---
    % 20MHz の HE-SU は 242-tone RU なので、ここで得られる CSI は 242 本。
    % Non-HT/HT(52本)・VHT(56本)とはサブキャリアの間隔も本数も異なる。
    demodHELTF = wlanHEDemodulate(pkt(ind.HELTF(1):ind.HELTF(2)), 'HE-LTF', cfgRx);
    heChanEst  = wlanHELTFChannelEstimate(demodHELTF, cfgRx);

    ofdmInfo = wlanHEOFDMInfo('HE-Data', cfgRx);
    info.numSubcarriers = size(heChanEst, 1);
    if isfield(ofdmInfo, 'ActiveFrequencyIndices') && ...
            numel(ofdmInfo.ActiveFrequencyIndices) == info.numSubcarriers
        info.subcIdx = double(ofdmInfo.ActiveFrequencyIndices(:)).';
    end

    % --- HE-Data の復号 ---
    % 雑音分散は L-LTF から求めた値 (noiseEst) をそのまま使っている。
    % HE-Data はサブキャリア間隔が 1/4 (78.125kHz) なので、厳密には
    % L-LTF 由来の値とスケールが一致しない。MMSE 等化と LDPC の LLR に
    % 効くため、「HE-SIG-A は解釈できるのに HE-Data の FCS が通らない」
    % 場合はここを最初に疑うとよい (HE-LTF から雑音を推定し直す)。
    demodData = wlanHEDemodulate(pkt(ind.HEData(1):ind.HEData(2)), 'HE-Data', cfgRx);
    [demodData, info.pilotTracked] = heTrackPilotErrorCompat(demodData, heChanEst, cfgRx, 'HE-Data');
    [eqData, csiData] = heEqualizeCompat(demodData(ofdmInfo.DataIndices, :, :), ...
        heChanEst(ofdmInfo.DataIndices, :, :), noiseEst, cfgRx, 'HE-Data');
    rxPSDU = heDataBitRecoverCompat(eqData, noiseEst, csiData, cfgRx);

    % --- HE の PSDU は必ず A-MPDU なので分解してから MPDU を復号 ---
    % wlanAMPDUDeaggregate / wlanMPDUDecode は wlanHERecoveryConfig ではなく
    % 送信用の wlanHESUConfig を受け取るため、復号したパラメータから作り直す。
    cfgMACPhy = buildHEMACConfig(cfgRx, info.psduLen, chanBW);
    [cfgMAC, payload, ok, deagInfo] = decodeAggregatedPSDU(rxPSDU, cfgMACPhy);
    info.mpduCount  = deagInfo.mpduCount;
    info.deagStatus = deagInfo.status;
    info.nonBinary  = deagInfo.nonBinary;
    info.dataClass  = deagInfo.dataClass;
    info.dataRange  = deagInfo.dataRange;
    info.mpduStatus = deagInfo.mpduStatus;
    info.hdrReason  = deagInfo.hdrReason;
    info.mpduBytes  = deagInfo.mpduBytes;
    if ~ok
        info.reason = sprintf('MPDU復号失敗(分解=%s,MPDU数=%d)', ...
            deagInfo.status, deagInfo.mpduCount);

        % MAC ヘッダは読めなかったが、CSI は HE-LTF から既に得られている。
        % HE-SIG-A の BSS Color は CRC を通っており、どの BSS のパケットかを
        % 示すので、これを手掛かりに後段で対象SSIDへ帰属させられる。
        % CSI を捨てないよう「CSIのみ」の記録として返す (status=2)。
        entry = struct( ...
            'timeSec',     0, ...
            'bssid',       '', ...        % 送信元不明
            'addr1',       '', ...
            'frameType',   '(MAC未復号)', ...
            'ssid',        '', ...
            'phyFormat',   'HE', ...
            'bssColor',    info.bssColor, ...
            'uplink',      info.uplink, ...
            'fcsVerified', false, ...
            'mpduCount',   0, ...
            'csi',         heChanEst(:).');
        status = 2;
        return;
    end

    entry = buildEntry(cfgMAC, payload, 'HE', heChanEst);
    if ~isempty(entry)
        entry.bssColor    = info.bssColor;
        entry.uplink      = info.uplink;
        entry.fcsVerified = ~deagInfo.headerOnly;
        entry.mpduCount   = deagInfo.mpduCount;   % 0 = A-MPDU分解失敗
    end
    status = 1;
end

function chanEst = preHEChannelEstimateCompat(lsigDemod, lltfChanEst, chanBW)
    % pre-HE フィールド (L-SIG / RL-SIG / HE-SIG-A) 用のチャネル推定を返す。
    %
    % HE PPDU ではこれらのフィールドが 20MHz あたり 56 本のサブキャリアで
    % 送られる (レガシーの 52 本 + 両端の 4 本)。両端 4 本を L-SIG/RL-SIG
    % から推定するのが wlanPreHEChannelEstimate。
    try
        chanEst = wlanPreHEChannelEstimate(lsigDemod, lltfChanEst, chanBW);
        return;
    catch
        % wlanPreHEChannelEstimate が無い MATLAB 向けのフォールバック
    end

    % フォールバック: L-LTF の推定値を両端に複製して本数だけ合わせる。
    % 追加される 4 本 (k=±27, ±28) は隣接サブキャリアと相関が高く、
    % SIG フィールドは BPSK + 畳み込み符号なのでこの近似で復号できる。
    info = wlanHEOFDMInfo('L-SIG', chanBW);
    h    = lltfChanEst(:, 1);
    nPre = double(info.NumTones);
    nL   = numel(h);
    if nPre <= nL
        chanEst = h(1:nPre);
        return;
    end
    nEdge   = floor((nPre - nL) / 2);
    chanEst = [repmat(h(1), nEdge, 1); h; repmat(h(end), nPre - nL - nEdge, 1)];
end

function snrDb = heRequiredSNR(mcs)
    % HE (20MHz, NSS=1) で MCS ごとにおおよそ必要な受信SNR [dB]。
    % IEEE 802.11ax の受信感度規定から導いた目安で、実装や環境で数dB動く。
    % 「復号できないのが受信品質のせいか、処理のせいか」を切り分ける用途。
    table = [2 5 9 11 15 18 20 25 29 31 34 37];   % MCS0..MCS11
    if mcs >= 0 && mcs <= 11
        snrDb = table(mcs + 1);
    else
        snrDb = NaN;
    end
end

function name = heModulationName(mcs)
    % MCS 番号に対応する変調方式と符号化率
    names = {'BPSK 1/2', 'QPSK 1/2', 'QPSK 3/4', '16QAM 1/2', '16QAM 3/4', ...
             '64QAM 2/3', '64QAM 3/4', '64QAM 5/6', '256QAM 3/4', '256QAM 5/6', ...
             '1024QAM 3/4', '1024QAM 5/6'};
    if mcs >= 0 && mcs <= 11
        name = names{mcs + 1};
    else
        name = '不明';
    end
end

function [sym, applied] = heTrackPilotErrorCompat(sym, chanEst, cfg, field)
    % パイロットサブキャリアを使って残留位相誤差 (CPE) と振幅誤差を補正する。
    %
    % なぜ必要か:
    %   精CFO推定 (wlanFineCFOEstimate) は L-LTF の 8us だけを見るので、
    %   数百 Hz 程度の残差が残る。さらに送受信のサンプリングクロック差
    %   (SFO) もある。どちらも「時間が経つほど位相が回る」誤差なので、
    %   プリアンブルや SIG フィールド (先頭 40us) では無視できても、
    %   A-MPDU で数 ms 続く HE-Data の後半では位相が数ラジアン回ってしまう。
    %   残留 CFO が 200 Hz でも 2ms 後には約14度回り、64QAM 以上は壊れる。
    %
    % sym は data+pilot 両方のサブキャリアを含む [Nst x Nsym x Nr] を渡すこと
    % (等化のために data だけ取り出す前に呼ぶ)。
    %
    % 失敗したフィールドだけを個別に無効化する点が重要。まとめて無効化すると、
    % 先頭の L-SIG で失敗しただけで、本当に効かせたい HE-Data の補正まで
    % 止まってしまう。
    persistent disabledFields
    if isempty(disabledFields)
        disabledFields = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    end

    key = char(string(field));
    if isKey(disabledFields, key) && disabledFields(key)
        applied = false;
        return;
    end

    try
        sym = wlanHETrackPilotError(sym, chanEst, cfg, field);
        applied = true;
    catch ME
        % 補正できなくても復号自体は続行できるので、そのフィールドについて
        % 一度だけ警告し、以降は追跡なしで進む
        % (毎パケット例外を出すと非常に遅くなる)。
        disabledFields(key) = true;
        applied = false;
        warning('decodeIQ_HE:noPilotTracking', ...
            ['%s の位相追跡を行えません: %s\n', ...
             '  このフィールドは補正なしで進みます。HE-Data で出ている場合、\n', ...
             '  長い A-MPDU の復号率が大きく下がります。\n', ...
             '  wlanHETrackPilotError は R2019b 以降の WLAN Toolbox に含まれます。'], ...
            key, ME.message);
    end
end

function [eqSym, csi] = heEqualizeCompat(sym, chEst, noiseEst, cfg, field)
    % HE のフィールドを等化する。wlanHEEqualize があればそれを使い、
    % 無ければ受信アンテナ1本 (SISO) 前提の MMSE 等化を自前で行う。
    try
        [eqSym, csi] = wlanHEEqualize(sym, chEst, noiseEst, cfg, field);
        return;
    catch
    end

    % フォールバック: SISO の MMSE 等化
    %   eq = conj(h)*y / (|h|^2 + N0),  CSI = |h|^2
    h = chEst;
    h = reshape(h, size(h, 1), []);
    h = h(:, 1);
    y = reshape(sym, size(sym, 1), []);   % [サブキャリア x シンボル]
    csi   = abs(h).^2;
    eqSym = (conj(h) .* y) ./ (csi + noiseEst);
end

function [failCheck, lsigInfo] = lsigBitRecoverCompat(eqLSIG, noiseEst, csiLSIG)
    % HE PPDU の L-SIG を復号し、パリティ結果と LENGTH/MCS を返す。
    % HE では L-SIG が RL-SIG として繰り返し送られるので 2 シンボルまとめて
    % 渡すと誤り耐性が上がるが、これを受け付けない版があるので、
    % その場合は L-SIG の 1 シンボルだけでもう一度試す。
    try
        [~, failCheck, lsigInfo] = wlanLSIGBitRecover(eqLSIG, noiseEst, csiLSIG);
        return;
    catch ME
        if size(eqLSIG, 2) < 2
            rethrow(ME);
        end
    end
    [~, failCheck, lsigInfo] = wlanLSIGBitRecover(eqLSIG(:, 1, :), noiseEst, csiLSIG);
end

function [bits, failCRC] = heSIGABitRecoverCompat(eqSIGA, noiseEst, csiSIGA)
    % HE-SIG-A の情報ビットを復元する。
    % HE-EXT-SU では HE-SIG-A が 4 シンボル (2 シンボルを 2 回繰り返し) に
    % なるが、これを受け付けない版があるので、その場合は先頭 2 シンボルだけで
    % もう一度試す。
    try
        [bits, failCRC] = wlanHESIGABitRecover(eqSIGA, noiseEst, csiSIGA);
        return;
    catch ME
        if size(eqSIGA, 2) <= 2
            rethrow(ME);
        end
    end
    [bits, failCRC] = wlanHESIGABitRecover(eqSIGA(:, 1:2, :), noiseEst, csiSIGA);
end

function rxPSDU = heDataBitRecoverCompat(eqData, noiseEst, csiData, cfgRx)
    % HE-Data のビットを復元する。csi を受け取らない版があるので両方試す。
    try
        rxPSDU = wlanHEDataBitRecover(eqData, noiseEst, csiData, cfgRx);
        return;
    catch
    end
    rxPSDU = wlanHEDataBitRecover(eqData, noiseEst, cfgRx);
end

function cfg = buildHEMACConfig(cfgRx, psduLen, chanBW)
    % A-MPDU 分解 / MPDU 復号に渡す PHY コンフィグを作る。
    % wlanAMPDUDeaggregate / wlanMPDUDecode は受信用の wlanHERecoveryConfig を
    % 受け付けないため、復号済みパラメータから送信用の wlanHESUConfig を作る。
    % (MAC の解析に効くのは「HE のフォーマットである」ことと帯域幅だけなので、
    %  細かい変調パラメータの設定に失敗しても致命的ではない)
    try
        cfg = wlanHESUConfig( ...
            'ChannelBandwidth',     getProp(cfgRx, 'ChannelBandwidth', chanBW), ...
            'NumTransmitAntennas',  1, ...
            'NumSpaceTimeStreams',  1, ...
            'MCS',                  double(getProp(cfgRx, 'MCS', 0)), ...
            'ChannelCoding',        char(string(getProp(cfgRx, 'ChannelCoding', 'LDPC'))), ...
            'GuardInterval',        double(getProp(cfgRx, 'GuardInterval', 0.8)), ...
            'HELTFType',            double(getProp(cfgRx, 'HELTFType', 4)), ...
            'APEPLength',           max(double(psduLen), 1));
        return;
    catch
    end

    try
        cfg = wlanHESUConfig('ChannelBandwidth', getProp(cfgRx, 'ChannelBandwidth', chanBW));
    catch
        cfg = wlanHESUConfig();
    end
end

function v = getProp(obj, name, defaultVal)
    % オブジェクトのプロパティを安全に読む (無ければ既定値)。
    % wlanHERecoveryConfig は PPDU 形式によって持つプロパティが変わるため。
    v = defaultVal;
    try
        p = obj.(name);
        if ~isempty(p)
            v = p;
        end
    catch
    end
end

function mhz = bwStringToMHz(bwStr)
    % 'CBW20' -> 20 のように WLAN Toolbox の帯域幅指定を MHz の数値へ直す
    s = char(string(bwStr));
    d = regexp(s, '\d+', 'match', 'once');
    if isempty(d)
        mhz = 0;
    else
        mhz = str2double(d);
    end
end

function [cfgMAC, payload, ok, deagInfo] = decodeAggregatedPSDU(rxPSDU, cfgFormat)
    % HT/VHT の PSDU を A-MPDU として分解し、最初に復号できた MPDU を返す。
    % cfgFormat は wlanHTConfig または wlanVHTConfig のいずれでもよい。
    cfgMAC = [];
    payload = [];
    ok = false;
    deagInfo = struct('status', 'N/A', 'mpduCount', 0, 'headerOnly', false, ...
        'nonBinary', false, 'dataClass', '', 'dataRange', [0 0]);
    % 切り分け用の計測。「なぜ MPDU が読めなかったのか」を
    % FCS の失敗なのか / MATLAB が非対応のフレーム種別なのか /
    % そもそもヘッダが壊れているのか、に分けて記録する。
    deagInfo.mpduStatus = {};   % wlanMPDUDecode が返したステータス
    deagInfo.hdrReason  = {};   % 自前ヘッダ解析の棄却理由
    deagInfo.mpduBytes  = [];   % 取り出せた MPDU の長さ [byte]

    mpduList = {};
    try
        [mpduList, ~, deagStatus] = wlanAMPDUDeaggregate(rxPSDU, cfgFormat, 'DataFormat', 'bits');
        deagInfo.status = char(string(deagStatus));
    catch ME
        deagInfo.status = ['例外: ' ME.message];
    end
    deagInfo.mpduCount = numel(mpduList);

    % 全体の status が Success でなくても、取り出せた MPDU は個別に試す
    % (一部のサブフレームだけ壊れている場合でも残りは復号できるため)
    %
    % 重要: wlanAMPDUDeaggregate に 'DataFormat','bits' を指定しても、返る
    % MPDU が16進文字列 (char, '0'-'9''A'-'F') になる場合がある。その形式の
    % まま 'DataFormat','bits' で wlanMPDUDecode を呼ぶと形式不一致で必ず
    % 失敗するため、実際のデータ形式を判定して合わせる。
    wState = warning('off', 'all');
    for m = 1:numel(mpduList)
        [dfmt, octPeek] = mpduDataFormat(mpduList{m});
        if isempty(dfmt)
            continue;   % 判別できない形式
        end
        deagInfo.mpduBytes(end+1) = numel(octPeek);
        try
            [c, p, st] = wlanMPDUDecode(mpduList{m}, cfgFormat, 'DataFormat', dfmt);
            deagInfo.mpduStatus{end+1} = char(string(st));
            if strcmpi(string(st), "Success")
                warning(wState);
                cfgMAC = c; payload = p; ok = true;
                return;
            end
        catch ME
            deagInfo.mpduStatus{end+1} = ['例外: ' ME.identifier];
        end
    end
    warning(wState);

    % A-MPDU として分解できなかった場合は単一 MPDU として試す
    if isempty(mpduList)
        dfmt = mpduDataFormat(rxPSDU);
        if ~isempty(dfmt)
            try
                wState = warning('off', 'all');
                [c, p, st] = wlanMPDUDecode(rxPSDU, cfgFormat, 'DataFormat', dfmt);
                warning(wState);
                if strcmpi(string(st), "Success")
                    cfgMAC = c; payload = p; ok = true;
                    return;
                end
            catch
            end
        end
    end

    % --- フォールバック: FCS 検証に失敗しても MAC ヘッダから送信元を読む ---
    % FCS は MPDU 全体 (最大 1500B 超) を検証するため、末尾の 1 ビット誤りでも
    % 失敗する。一方 CSI の帰属に必要なのは先頭 24 バイトのヘッダだけであり、
    % ここが無傷である確率は高い。
    %
    % ただし対象は A-MPDU 分解に成功した MPDU に限る。分解に失敗した PSDU
    % 全体をヘッダとみなす処理は、デリミタで区切れていない = 復号が壊れて
    % いる証拠があるにもかかわらず、緩い妥当性検査を偶然通ったランダムな
    % ビット列を「送信元アドレス」として大量に生んでしまうため行わない。
    for m = 1:numel(mpduList)
        [dfmt, oct] = mpduDataFormat(mpduList{m});
        if isempty(dfmt)
            % ビット列でも16進文字列でもない = 想定外のデータ形式。
            % 静かに壊れた結果を出さないよう記録して弾く。
            raw = double(mpduList{m}(:));
            deagInfo.nonBinary = true;
            deagInfo.dataClass = class(mpduList{m});
            if ~isempty(raw)
                deagInfo.dataRange = [min(raw) max(raw)];
            end
            continue;
        end
        hdr = parseMACHeaderFromOctets(oct);
        if hdr.valid
            cfgMAC  = hdr;      % wlanMACFrameConfig ではなく自前の構造体
            payload = [];
            ok = true;
            deagInfo.headerOnly = true;
            return;
        end
        deagInfo.hdrReason{end+1} = hdr.reason;
    end
end

function [fmtStr, oct] = mpduDataFormat(data)
    % MPDU データの実際の形式を判定し、オクテット列も返す。
    %   fmtStr: 'bits'   … 0/1 の数値ベクトル
    %           'octets' … 16進文字列 (char/string)
    %           ''       … 判別不能
    % wlanMPDUDecode / wlanAMPDUDeaggregate の 'DataFormat' にはこの値を
    % そのまま渡せる。
    fmtStr = '';
    oct    = [];

    if isempty(data)
        return;
    end

    if ischar(data) || isstring(data)
        s = char(data);
        s = s(:).';
        s = s(~isspace(s));
        if isempty(s) || ~all(isstrprop(s, 'xdigit'))
            return;   % 16進文字列ではない
        end
        fmtStr = 'octets';
        oct    = hexStrToOctets(s);
        return;
    end

    d = double(data(:));
    if all(d == 0 | d == 1)
        fmtStr = 'bits';
        oct    = bitsToOctets(d);
    end
end

function oct = hexStrToOctets(s)
    % 16進文字列 ('4A3B...') をオクテット列 [74 59 ...] へ変換する
    n = floor(numel(s) / 2) * 2;
    if n == 0
        oct = [];
        return;
    end
    v = sscanf(s(1:n), '%2x');
    oct = v(:).';
end

function hdr = parseMACHeaderFromOctets(oct)
    % MPDU のオクテット列から MAC ヘッダ (Frame Control / Address1-3) を
    % 直接読む。FCS を検証しないため、内容が正しい保証は無い点に注意。
    hdr = struct('valid', false, 'FrameType', '', 'Address1', '', 'Address2', '', ...
        'ManagementConfig', [], 'headerOnly', true, 'reason', '');

    if numel(oct) < 24
        hdr.reason = sprintf('MPDUが%dバイトしかない(24バイト未満)', numel(oct));
        return;   % Address2 まで読めない
    end
    if any(oct < 0 | oct > 255 | mod(oct, 1) ~= 0)
        hdr.reason = 'オクテット範囲外の値';
        return;
    end

    fc0     = oct(1);
    fc1     = oct(2);
    version = bitand(fc0, 3);
    type    = bitand(bitshift(fc0, -2), 3);
    subtype = bitand(bitshift(fc0, -4), 15);

    if version ~= 0
        hdr.reason = 'ProtocolVersion≠0';
        return;
    end

    % ToDS/FromDS の組み合わせ検査。ToDS=1&FromDS=1 は 4アドレス形式
    % (WDS) で、通常のインフラストラクチャ BSS では使われない。
    % また管理フレームでは両方 0 でなければならない。
    toDS   = bitand(fc1, 1);
    fromDS = bitand(bitshift(fc1, -1), 1);
    if toDS == 1 && fromDS == 1
        hdr.reason = 'ToDS=1かつFromDS=1(4アドレス形式)';
        return;
    end
    if type == 0 && (toDS ~= 0 || fromDS ~= 0)
        hdr.reason = '管理フレームなのにToDS/FromDS≠0';
        return;
    end

    % Address2 を持たないフレーム (ACK=1101, CTS=1100 の制御フレーム) は対象外
    if type == 1 && any(subtype == [12 13])
        hdr.reason = 'ACK/CTS(Address2なし)';
        return;
    end

    switch type
        case 0
            typeName = 'Management';
            if subtype == 8
                typeName = 'Beacon';
            elseif subtype == 5
                typeName = 'Probe Response';
            end
        case 1
            typeName = 'Control';
        case 2
            typeName = 'Data';
        otherwise
            hdr.reason = 'Type=3(予約値)';
            return;
    end

    addr1 = oct(5:10);
    addr2 = oct(11:16);

    % 送信元 (Address2) が全0 / ブロードキャストのものは復号が壊れている。
    % また Address2 のグループビット (先頭オクテットの bit0) は送信元
    % アドレスでは必ず 0 になる。
    % 3つの条件を別々に記録する。特に「全0」が多い場合は、LDPC 復号が
    % 収束せず全零符号語 (これは常に妥当な符号語なので、復号失敗時の
    % 行き先になりやすい) を返している証拠になる。= 受信品質の問題であり、
    % 復号器の実装ミスではない。
    if all(addr2 == 0)
        hdr.reason = 'Address2が全0(LDPCが全零符号語に落ちた可能性)';
        return;
    end
    if all(addr2 == 255)
        hdr.reason = 'Address2が全1';
        return;
    end
    if bitand(addr2(1), 1) == 1
        hdr.reason = 'Address2のグループビットが1(送信元アドレスではあり得ない)';
        return;
    end

    hdr.valid     = true;
    hdr.FrameType = typeName;
    hdr.Address1  = upper(sprintf('%02X', addr1));
    hdr.Address2  = upper(sprintf('%02X', addr2));
end

function ndbps = vhtNDBPS20MHz(mcs)
    % VHT 20MHz / 1空間ストリーム の NDBPS (1 OFDM シンボルあたりのデータビット数)
    % NSD=52, NDBPS = 52 * NBPSCS * 符号化率
    table = [26 52 78 104 156 208 234 260 312];   % MCS0..MCS8
    if mcs >= 0 && mcs <= 8
        ndbps = table(mcs + 1);
    else
        ndbps = [];   % MCS9 は 20MHz/1SS では未定義
    end
end

function entry = buildEntry(cfgMAC, payload, phyFormat, chanEst)
    % 復号済み MAC 情報と CSI から記録用の構造体を作る。
    % Address2 (送信元アドレス) を持たないフレームは BSSID 判定に使えない
    % ため空を返す。
    %
    % 'bssid' には Address2 (送信元/TA) を入れる。ダウンリンク (AP→クライアント)
    % では Address2=APのBSSIDだが、アップリンク (クライアント→AP) では
    % Address2=クライアントのMACになり、代わりに Address1(宛先/RA)=APのBSSID
    % となる。どちらの向きも取りこぼさないよう 'addr1' も別途保持する。
    entry = [];
    frameType = char(string(cfgMAC.FrameType));

    % ACK と CTS は Address1 のみを持ち Address2 が無い。
    % (この場合 cfgMAC.Address2 には既定値が残るため BSSID として使えない)
    if any(strcmpi(frameType, {'ACK', 'CTS'}))
        return;
    end

    ssidStr = '';
    if any(strcmpi(frameType, {'Beacon', 'Probe Response'}))
        ssidStr = extractSSID(cfgMAC, payload);
    end

    addr1Str = '';
    try
        addr1Str = char(string(cfgMAC.Address1));
    catch
        % Address1 を持たないオブジェクトの場合は空のままにする
    end

    entry = struct( ...
        'timeSec',     0, ...            % 呼び出し側で設定
        'bssid',       char(string(cfgMAC.Address2)), ...
        'addr1',       addr1Str, ...
        'frameType',   frameType, ...
        'ssid',        ssidStr, ...
        'phyFormat',   phyFormat, ...
        'bssColor',    -1, ...           % HE のみ。呼び出し側で HE-SIG-A の値を入れる
        'uplink',      false, ...        % HE のみ。HE-SIG-A の UL/DL ビット
        'fcsVerified', true, ...         % HT/VHT のヘッダのみ復号時は呼び出し側で false
        'mpduCount',   1, ...            % 集約されている場合は呼び出し側で上書き
        'csi',         chanEst(:).');
end

function ssidStr = extractSSID(cfgMAC, payload)
    % Beacon/Probe Response から SSID を取得する。
    % wlanMPDUDecode は管理フレームの内容を cfgMAC.ManagementConfig
    % (wlanMACManagementConfig) に格納するため、まずそこから読む。
    ssidStr = '';
    try
        mgmt = cfgMAC.ManagementConfig;
        if ~isempty(mgmt)
            s = char(string(mgmt.SSID));
            if ~isempty(s)
                ssidStr = s;
                return;
            end
        end
    catch
        % ManagementConfig を持たない/参照できない場合は下のフォールバックへ
    end

    % フォールバック: フレームボディを自前で解析する
    ssidStr = parseSSIDFromMgmtFrame(payload);
end

function [mcs, psduLen] = decodeLSIGBits(bits)
    % 802.11a/g L-SIG (24bit: RATE(4)+Reserved(1)+LENGTH(12)+Parity(1)+Tail(6))
    % を解釈し、Non-HT MCS インデックス (0..7) と PSDU長 [byte] を返す。
    % RATE の対応は IEEE Std 802.11-2016 Table 17-6 による。
    bits = double(bits(:)).';
    if numel(bits) < 17
        mcs = []; psduLen = 0;
        return;
    end

    rateKey = sprintf('%d', bits(1:4));
    rateMap = containers.Map( ...
        {'1101','1111','0101','0111','1001','1011','0001','0011'}, ...
        {0,      1,     2,     3,     4,     5,     6,     7});
    if ~isKey(rateMap, rateKey)
        mcs = []; psduLen = 0;
        return;
    end
    mcs = rateMap(rateKey);

    lenBits = bits(6:17);   % LENGTH は LSB 先頭
    psduLen = sum(lenBits .* 2.^(0:11));
end

function out = parseVHTSIGABits(bits)
    % VHT-SIG-A (48bit = SIG-A1(24) + SIG-A2(24)) を解釈する。
    % IEEE Std 802.11ac-2013, Table 22-12 に基づく。多ビットフィールドは
    % LSB 先頭で送信される。
    %   bits(1:2)   BW (00=20MHz)          bits(4)     STBC
    %   bits(5:10)  Group ID               bits(11:13) NSTS-1
    %   bits(25)    Short GI               bits(26)    Short GI NSYM 曖昧性解消
    %   bits(27)    Coding (0=BCC,1=LDPC)  bits(28)    LDPC Extra OFDM Symbol
    %   bits(29:32) SU VHT-MCS
    bits = double(bits(:)).';
    out = struct('isValid', false, 'is20MHz', false, 'isSU', false, ...
        'nsts', 0, 'stbc', false, 'groupId', 0, 'gi', 0, 'coding', 0, ...
        'mcs', 0, 'bwMHz', 0, 'reservedOK', false, ...
        'sgiDisamb', 0, 'ldpcExtra', 0);

    if numel(bits) < 34
        return;
    end

    % 自己チェック: 仕様上 1 に固定されている予約ビット
    %   SIG-A1 B2  -> bits(3),  SIG-A1 B23 -> bits(24),  SIG-A2 B9 -> bits(34)
    % これが揃わない場合、本関数のビット位置の想定がズレている。
    out.reservedOK = (bits(3) == 1) && (bits(24) == 1) && (bits(34) == 1);

    bw       = bits(1:2);
    stbcBit  = bits(4);
    groupId  = bits(5:10);
    nstsBits = bits(11:13);

    bwTable     = [20 40 80 160];
    out.bwMHz   = bwTable(sum(bw .* 2.^(0:1)) + 1);
    out.is20MHz = all(bw == 0);
    % SU-VHT は Group ID = 0 または 63 (全ビット0 または 全ビット1)
    out.isSU    = all(groupId == 0) || all(groupId == 1);
    if all(nstsBits == 0)
        out.nsts = 1;   % 全ビット0ならビット順によらず NSTS=1 と確定できる
    else
        out.nsts = 1 + sum(nstsBits .* 2.^(0:2));
    end
    out.stbc    = stbcBit == 1;
    out.groupId = sum(groupId .* 2.^(0:5));
    out.gi        = bits(25);
    out.sgiDisamb = bits(26);
    out.coding    = bits(27);
    out.ldpcExtra = bits(28);
    out.mcs       = sum(bits(29:32) .* 2.^(0:3));
    out.isValid   = true;
end

function apepLen = decodeVHTSIGBLength(bits)
    % VHT-SIG-B (20MHz, SU) の Length フィールド(17bit, LSB先頭)を解釈し、
    % APEP長 [byte] を返す。20MHz では 4 オクテット単位で符号化されるため
    % 4 倍する。※この換算は実機未検証で、VHT が復号できない場合に
    %   最初に見直すべき箇所。
    bits = double(bits(:)).';
    if numel(bits) < 17
        apepLen = [];
        return;
    end
    apepLen = sum(bits(1:17) .* 2.^(0:16)) * 4;
end

function ssidStr = parseSSIDFromMgmtFrame(payload)
    % Beacon/Probe Response のフレームボディから先頭の
    % Information Element (SSID, Element ID = 0) を抽出する。
    % 固定フィールド: Timestamp(8) + BeaconInterval(2) + CapabilityInfo(2) = 12byte
    ssidStr = '';
    if isempty(payload)
        return;
    end
    if iscell(payload)
        payload = payload{1};
    end
    if isempty(payload)
        return;
    end

    % payload はビット列でも16進文字列でも返り得るので、実際の形式を
    % 判定してオクテット列へ揃える。
    [dfmt, oct] = mpduDataFormat(payload);
    if isempty(dfmt)
        return;
    end
    payload = uint8(oct);

    if numel(payload) < 14
        return;
    end
    ieOffset = 12;
    elemId  = payload(ieOffset + 1);
    elemLen = payload(ieOffset + 2);
    if elemId ~= 0
        return;   % 先頭 IE が SSID でない
    end
    if numel(payload) < ieOffset + 2 + elemLen || elemLen == 0
        return;
    end
    ssidStr = char(payload(ieOffset + 3 : ieOffset + 2 + elemLen));
end

function bytes = bitsToOctets(bits)
    % ビット列をオクテット列へ変換する (各オクテット内は LSB 先頭)。
    %
    % 入力が 0/1 でない場合 (軟判定値・オクテット列など) は変換結果が
    % 0..255 の範囲を外れ、MAC アドレスとして解釈すると 6バイトのはずが
    % 12バイト分の文字列になるなど、静かに壊れたデータを生む。そのため
    % ここで明示的に検査し、2値でなければ空を返して呼び出し側で弾く。
    bits = double(bits(:));
    if isempty(bits) || ~all(bits == 0 | bits == 1)
        bytes = [];
        return;
    end
    n = floor(numel(bits) / 8) * 8;
    if n == 0
        bytes = [];
        return;
    end
    b = reshape(bits(1:n), 8, []);
    bytes = (2.^(0:7)) * b;   % 1 x (n/8), 各要素は 0..255
end

function F = buildFrameLog(pktLog, targetBSSID)
    % 復号できた全フレームの時刻と送信元・宛先を、数値だけの表にまとめる。
    % IMU 計測アプリが送る Wi-Fi の合図 (ノート PC → AP の上りデータフレーム、
    % 一定間隔の小さなかたまり) を後で探すのに使う。中身は暗号化で読めないが、
    % 送信元・宛先のアドレスと時刻は読める。
    n = numel(pktLog);
    F = struct();
    F.description = ['all decoded frames (any BSS). ta/ra = transmitter/receiver MAC as 48-bit ', ...
                     'numbers (NaN = unknown). phy: 1=Non-HT 2=HT 3=VHT 4=HE. isData: frame type ', ...
                     'contains "Data". toAP: receiver is the target BSSID. timeSec: seconds from capture start.'];
    F.timeSec     = zeros(n, 1);
    F.ta          = nan(n, 1);
    F.ra          = nan(n, 1);
    F.phy         = zeros(n, 1);
    F.isData      = false(n, 1);
    F.fcsVerified = false(n, 1);
    F.uplinkHE    = false(n, 1);
    F.targetBSSID = macToNum(targetBSSID);
    phyNames = {'Non-HT', 'HT', 'VHT', 'HE'};
    for k = 1:n
        e = pktLog(k);
        F.timeSec(k) = e.timeSec;
        F.ta(k) = macToNum(e.bssid);      % pktLog の bssid は Address2 (送信元)
        F.ra(k) = macToNum(e.addr1);
        p = find(strcmpi(phyNames, e.phyFormat), 1);
        if ~isempty(p)
            F.phy(k) = p;
        end
        F.isData(k)      = ~isempty(strfind(lower(char(string(e.frameType))), 'data'));
        F.fcsVerified(k) = logical(e.fcsVerified);
        F.uplinkHE(k)    = logical(e.uplink);
    end
    F.toAP = ~isnan(F.ra) & (F.ra == F.targetBSSID);
end

function v = macToNum(s)
    % 'AA:BB:CC:DD:EE:FF' や 'AABBCCDDEEFF' を 48 bit の数値にする (double で正確に表せる)
    v = NaN;
    if isempty(s)
        return;
    end
    h = regexprep(char(string(s)), '[^0-9A-Fa-f]', '');
    if numel(h) ~= 12
        return;
    end
    v = hex2dec(h);
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
