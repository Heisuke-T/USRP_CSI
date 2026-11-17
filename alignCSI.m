%% alignCSI.m
% =========================================================================
%  2 台の受信機 (antenna1 / antenna2) の CSI をパケット単位で対応づけ、
%  時間軸を揃えて、両方がそろっている区間から 80 秒 (既定) を切り出して保存する
% -------------------------------------------------------------------------
%  使う場面:
%    2 台の USRP をそれぞれ別の PC で captureIQ.m などで回すと、開始時刻が
%    数秒ずれる。また各 PC の時計も合っていないので、記録された時刻からは
%    揃えられない。
%
%  揃え方 (パケットの到着時刻の突き合わせ):
%    2 台は同じ AP・同じクライアントの同じパケットを受信している。同じパケット
%    なら、2 台での到着時刻の差 (= 2 台の時計のずれ) は 1 us 以下の精度で
%    そろう。一方、iperf3 で飽和した通信は数 ms 周期でほぼ同じ形が繰り返される
%    ので、粗い時間分解能で比べると「1 周期ずれた位置」とも区別がつかない。
%    違うパケット同士はバックオフ (9 us 単位のランダムな待ち時間) の分だけ
%    ずれるので、10 us の細かさで比べれば本当の位置だけが際立つ。これを使って、
%      1. 粗い位置合わせ   antenna1 の 5 か所 (各 1 秒) のパケットについて、
%                          antenna2 の同じ種類のパケットとの時刻差を
%                          ±maxOffsetSec の範囲で全部求め、10 us の幅に
%                          一番多く集まる時刻差を採る
%      2. ずれの追跡      1 秒ごとに、前の区間のずれの近く (±20 ms、見つから
%                          なければ ±2 s) で同じように求め直す。これでクロック
%                          のわずかな差 (数 ppm) や、オーバーランによる時刻の
%                          段差にも追従できる
%      3. パケットの対応づけ 同じ種類のパケット同士 (HE の下り同士、RTS 同士、
%                          Block Ack 同士 …) を 1 対 1 で対応づける
%      4. 検証と絞り込み  対応づけた組を、前後の組から求めた局所的なずれとの
%                          残差 (許容幅 ±pairToleranceUs) と、中身 (BSS Color・
%                          フレーム種別) で確かめ、外れたものを除く。
%                          オーバーランによる時刻の段差を見つけ、どちらの
%                          受信機でサンプルが抜けたかを判定する
%    の順に揃える。精度は 1 us 以下 (パケット間隔は数 ms)。
%
%    すでに復号済みの *_CSI.mat だけを使うので、復号し直す必要はない。
%
%  時間軸:
%    antenna1 (refTag) の発振器の時刻を基準にする。antenna1 でオーバーランに
%    よりサンプルが抜けた箇所は、抜けた分を時間軸に空白として戻す (その間の
%    パケットは antenna2 にだけある)。antenna2 の時刻はこれに変換する。
%    出力の timeSec* は、切り出し区間の先頭を 0 とする 2 台共通の時刻。
%
%  入力 (csiPath 内のペアを自動で探す):
%    S2_C_antenna1_WAX202_CSI.mat  と  S2_C_antenna2_WAX202_CSI.mat
%    (ファイル名の refTag を otherTag に置き換えた名前を相手とみなす)
%
%  出力 (受信機ごとに 1 つ):
%    S2_C_antenna1_WAX202_aligned_CSI.mat
%    S2_C_antenna2_WAX202_aligned_CSI.mat
%      中身の変数名は decodeIQ_HE.m の出力と同じ (csiHE / timeSecHE / ...)。
%      ResultCSI.m や SHARP-USRP の変換スクリプトにそのまま渡せる。
%
%      commonPacketsOnly = true (既定) のとき:
%        - 2 台が両方受けたパケットだけを、2 つのファイルで同じ順番に並べる。
%          antenna1 の i 番目と antenna2 の i 番目は同じパケット (同じ送信) で、
%          timeSec* は 2 つのファイルで完全に一致する。パケット番号で処理する
%          後段 (SHARP-USRP の位相サニタイズ・ドップラー・データセット作成) でも、
%          2 台は常に同じパケット・同じ時刻を指す
%        - 片方だけが受けた HE パケットは unpairedHE に、どちらが受けたかの
%          並び (実測の欠損パターン) は csiMeta.alignment.lossTraceHE に残す
%      commonPacketsOnly = false のとき:
%        - 各受信機が受けたパケットを全部残し、相手のファイルでの行番号を
%          pairIndex* に入れる (相手が受けていなければ 0)
%      共通:
%        - heLink で HE の上り/下りを選ぶ (既定は下り = AP からの送信だけ)。
%          上りと下りは送信機が違い CSI も別物なので、混ぜない
%        - CSI が全部 0 / NaN のパケットは除く (後段で片方だけ消えないように)
%        - timeSecLocal* に元の (各受信機の) 時刻を残す
%        - csiMeta.alignment に、揃えたときの情報と検証の結果を残す
%        - frameLog (decodeIQ_HE.m が保存する全フレームの記録) があれば引き継ぎ、
%          frameLog.timeSecAligned にこのファイルの時間軸 (切り出し区間の先頭 = 0) での
%          時刻を足す。IMU 計測アプリの Wi-Fi の合図を探して時刻を合わせるのに使う
%          (切り出し区間の外のフレームも残す。時刻は区間の外でも同じ規則で変換)
%    ログ: align_<日時>.log と align_<日時>_summary.csv
%
%  注意:
%    - 2 台が共通のパケットをある程度受けている必要がある (目安 3 割以上)。
%      片方がほとんど受けていない場合は失敗として報告し、出力は作らない。
%    - 開始時刻のずれが maxOffsetSec (既定 20 秒) を超える場合は見つけられない。
%      そのときは maxOffsetSec を大きくする。
% =========================================================================

clear; clc;

%% ------------------------------------------------------------------------
%  1. ユーザ設定パラメータ
%  ------------------------------------------------------------------------
cfg = struct();

% --- CSI ファイルのあるフォルダ (decodeIQ_HE.m の hddSavePath) ------------
cfg.csiPath = 'D:\IQ_csi';

% --- ペアの見分け方 ------------------------------------------------------
%     ファイル名の refTag を otherTag に置き換えたものを相手とみなす。
%     時間軸は refTag 側の時計を基準にする。
cfg.refTag   = 'antenna1';
cfg.otherTag = 'antenna2';

% --- 切り出す長さ [s] ----------------------------------------------------
cfg.outputDuration = 80;

% --- 切り出す位置 --------------------------------------------------------
%     'center' … 2 台がそろっている区間の真ん中から切り出す (前後を均等に捨てる)
%     'start'  … 2 台がそろった瞬間から切り出す
cfg.windowPosition = 'center';

% --- 出力するパケット ----------------------------------------------------
%     true  … 2 台が両方受けたパケットだけを、2 つのファイルで同じ順番に並べる。
%             i 番目どうしが同じパケットで、timeSec* も完全に一致する。
%             サイト間の対応づけ・補正や、2 台のドップラーを重ねる用途はこちら
%     false … 各受信機が受けたパケットを全部残す (相手の行番号は pairIndex*)
cfg.commonPacketsOnly = true;

% --- HE の上り/下り ------------------------------------------------------
%     'downlink' … AP -> クライアント (iperf3 のデータ) だけ
%     'uplink'   … クライアント -> AP だけ
%     'both'     … 両方 (送信機の違う 2 系列が 1 本に混ざる。以前の動作)
cfg.heLink = 'downlink';

% --- 対応づけの許容幅 [us] -----------------------------------------------
%     同じパケットの 2 台での時刻は 0.1〜0.3 us でそろう。同じ種類の別の
%     パケットは数十 us 以上離れているので、この幅で取り違えは起きない。
cfg.pairToleranceUs = 2;

% --- 2 台の開始時刻のずれとして想定する最大値 [s] ------------------------
cfg.maxOffsetSec = 20;

% --- すでに出力があるペアを飛ばすか --------------------------------------
%     古い版の alignCSI.m で作った出力は、true でも作り直す。
cfg.skipExisting = true;

%% ------------------------------------------------------------------------
%  2. 実行
%  ------------------------------------------------------------------------
runAlign(cfg);

%% ------------------------------------------------------------------------
%  ローカル関数
%  ------------------------------------------------------------------------
function runAlign(cfg)
    if ~exist(cfg.csiPath, 'dir')
        error('alignCSI:noFolder', 'フォルダがありません: %s', cfg.csiPath);
    end
    if ~any(strcmpi(cfg.heLink, {'downlink', 'uplink', 'both'}))
        error('alignCSI:badLink', 'heLink は downlink / uplink / both のどれかです: %s', cfg.heLink);
    end

    runTag  = datestr(now, 'yyyymmdd_HHMMSS');
    logFile = fullfile(cfg.csiPath, ['align_' runTag '.log']);
    csvFile = fullfile(cfg.csiPath, ['align_' runTag '_summary.csv']);
    diary(logFile);
    diaryCleanup = onCleanup(@() diary('off')); %#ok<NASGU>

    pairs = findPairs(cfg);
    if isempty(pairs)
        fprintf('ペアが見つかりませんでした (%s 内の *%s*_CSI.mat と *%s*_CSI.mat)。\n', ...
            cfg.csiPath, cfg.refTag, cfg.otherTag);
        return;
    end

    fprintf('時間軸を揃えるペア (基準: %s、HE: %s、%s):\n', cfg.refTag, linkName(cfg.heLink), ...
        ternary(cfg.commonPacketsOnly, '2 台に共通のパケットだけ出力', '全パケットを出力'));
    for k = 1:numel(pairs)
        if pairs(k).skip
            mark = '済';
        elseif pairs(k).outdated
            mark = '旧';
        else
            mark = '  ';
        end
        fprintf('  %s  %s\n', mark, pairs(k).label);
    end
    if any([pairs.outdated] & ~[pairs.skip])
        fprintf('  (旧 = 古い版の alignCSI.m の出力があるため作り直す)\n');
    end

    results = newResult(pairs(1));
    results(:) = [];
    for k = 1:numel(pairs)
        if pairs(k).skip
            res = newResult(pairs(k));
            res.status = 'SKIP';
        else
            try
                res = alignPair(pairs(k), cfg);
            catch ME
                res = newResult(pairs(k));
                res.status = 'FAIL';
                res.message = ME.message;
                if ~isempty(ME.stack)
                    % 想定外のエラーは、どこで起きたかも残す
                    res.message = sprintf('%s (%s の %d 行目)', ME.message, ...
                        ME.stack(1).name, ME.stack(1).line);
                end
                fprintf(2, '\n  失敗しました: %s\n', res.message);
            end
        end
        results(end+1) = res; %#ok<AGROW>
        writeAlignCsv(csvFile, results);
    end

    printAlignSummary(results);
    fprintf('\nログ       : %s\n', logFile);
    fprintf('結果の一覧 : %s\n', csvFile);
end

function v = alignVersion()
    % 出力の形式の版。これより古い出力は skipExisting = true でも作り直す。
    %   1: 時刻だけを揃えた (含まれるパケットは受信機ごとに違う)
    %   2: パケット単位で対応づけ、検証した。共通のパケットを同じ順番で出力
    v = 2;
end

function tf = isCurrentOutput(file)
    % 今の版の alignCSI.m で作った出力か
    tf = false;
    if ~isfile(file)
        return;
    end
    try
        S  = load(file, 'csiMeta');
        tf = S.csiMeta.alignment.version >= alignVersion();
    catch
        tf = false;
    end
end

function pairs = findPairs(cfg)
    lst   = dir(fullfile(cfg.csiPath, '*_CSI.mat'));
    names = {lst.name};
    names = names(~contains(names, '_aligned_CSI.mat'));   % 出力は入力にしない
    names = sort(names);

    pairs = struct('refFile', {}, 'otherFile', {}, 'refOut', {}, ...
        'otherOut', {}, 'label', {}, 'skip', {}, 'outdated', {});
    for k = 1:numel(names)
        if ~contains(names{k}, cfg.refTag)
            continue;
        end
        partner = strrep(names{k}, cfg.refTag, cfg.otherTag);
        if ~any(strcmp(names, partner))
            fprintf('  相手が見つからないため飛ばします: %s (探した名前: %s)\n', ...
                names{k}, partner);
            continue;
        end
        refOut   = fullfile(cfg.csiPath, regexprep(names{k}, '_CSI\.mat$', '_aligned_CSI.mat'));
        otherOut = fullfile(cfg.csiPath, regexprep(partner,  '_CSI\.mat$', '_aligned_CSI.mat'));
        current  = isCurrentOutput(refOut) && isCurrentOutput(otherOut);
        pr = struct( ...
            'refFile',   fullfile(cfg.csiPath, names{k}), ...
            'otherFile', fullfile(cfg.csiPath, partner), ...
            'refOut',    refOut, ...
            'otherOut',  otherOut, ...
            'label',     sprintf('%s <-> %s', names{k}, partner), ...
            'skip',      cfg.skipExisting && current, ...
            'outdated',  ~current && (isfile(refOut) || isfile(otherOut)));
        pairs(end+1) = pr; %#ok<AGROW>
    end
end

function res = newResult(pr)
    res = struct('label', pr.label, 'status', '', 'message', '', ...
        'startDelaySec', NaN, 'confidence', NaN, 'nPairs', 0, ...
        'residMaxUs', NaN, 'minSpacingUs', NaN, 'nRejected', 0, 'nMismatch', 0, ...
        'stepsRef', 0, 'stepsOther', 0, 'overrunRef', NaN, 'overrunOther', NaN, ...
        'driftPpm', NaN, 'overlapSec', NaN, 'outputSec', NaN, ...
        'nHECommon', 0, 'nHEOnlyRef', 0, 'nHEOnlyOther', 0, ...
        'ratioRef', NaN, 'ratioOther', NaN);
end

function res = alignPair(pr, cfg)
    res = newResult(pr);
    fprintf('\n=========================================================================\n');
    fprintf(' %s\n', pr.label);
    fprintf('=========================================================================\n');

    A = load(pr.refFile);
    B = load(pr.otherFile);
    evA = collectEvents(A);
    evB = collectEvents(B);
    [evA, evB] = indexKinds(evA, evB);
    D1  = max([captureLength(A); evA.t]);
    D2  = max([captureLength(B); evB.t]);
    res.overrunRef   = overrunCountOf(A);
    res.overrunOther = overrunCountOf(B);
    fprintf('  パケット数 (対象SSIDの全形式): %s %d 件 / %s %d 件\n', ...
        cfg.refTag, numel(evA.t), cfg.otherTag, numel(evB.t));
    fprintf('  キャプチャ時のオーバーラン  : %s %g 回 / %s %g 回\n', ...
        cfg.refTag, res.overrunRef, cfg.otherTag, res.overrunOther);
    if numel(evA.t) < 200 || numel(evB.t) < 200
        res = failResult(res, sprintf( ...
            'パケットが少なすぎて揃えられません (%d 件 / %d 件)。', numel(evA.t), numel(evB.t)));
        return;
    end

    % --- 1. 粗い位置合わせ ---
    co = coarseOffset(evA, evB, D1, cfg.maxOffsetSec);
    if isempty(co.conf)
        res = failResult(res, '位置合わせに使える区間がありません (パケットが少なすぎます)。');
        return;
    end
    fprintf('  粗い位置合わせ (%s の 1 秒区間ごとに、±%g 秒の範囲を 10 us の細かさで探索):\n', ...
        cfg.refTag, cfg.maxOffsetSec);
    for s = 1:numel(co.conf)
        fprintf('    %5.1f s 付近: ずれ %+.6f s, 一致 %4d / %4d 件, 確信度 %5.1f\n', ...
            co.segCenter(s), co.offset(s), co.count(s), co.nRef(s), co.conf(s));
    end
    [conf, best] = max(co.conf);
    L0 = co.offset(best);
    res.confidence = conf;
    if conf < 3
        res = failResult(res, sprintf( ...
            ['パケットの並びが一致する位置をはっきり見つけられませんでした (確信度 %.1f)。', ...
             '開始時刻のずれが %g 秒を超えている、または 2 台で共通に受けた', ...
             'パケットが少ない可能性があります。'], conf, cfg.maxOffsetSec));
        return;
    end

    % --- 2, 3. ずれの追跡と、パケットの対応づけ ---
    [ia, ib] = fineMatch(evA, evB, D1, co.segCenter(best), L0);
    if numel(ia) < 100
        res = failResult(res, sprintf( ...
            '対応づけられたパケットが %d 組しかなく、揃えられません。', numel(ia)));
        return;
    end

    % --- 4. 対応づけの検証と絞り込み、共通の時刻 ---
    [pp, ver] = verifyPairs(A, B, evA, evB, ia, ib, cfg.pairToleranceUs * 1e-6);
    res.nPairs       = ver.nPairs;
    res.residMaxUs   = ver.residMaxUs;
    res.minSpacingUs = min(ver.minSpacingRefUs, ver.minSpacingOtherUs);
    res.nRejected    = ver.nRejectedTol + ver.nIsolated;
    res.nMismatch    = ver.content.nColorMismatch + ver.content.nTypeMismatch;
    res.stepsRef     = numel(ver.stepRefSec);
    res.stepsOther   = numel(ver.stepOtherSec);
    res.driftPpm     = ver.driftPpm;
    if ver.nPairs < 100
        res = failResult(res, sprintf( ...
            '検証を通った対応が %d 組しかなく、揃えられません。', ver.nPairs));
        return;
    end

    % 各受信機の時刻 -> 共通の時刻 (対応づけたパケットを節点にする)
    tkA = evA.t(pp.ia);
    tkB = evB.t(pp.ib);
    % サンプルが抜けた所では、抜けの直前と直後の節点の間にある片方だけの
    % パケットが、抜けの前後どちらだったかが局所の時刻からは分からない。
    % 抜けていた間は相手だけが受けるので、相手のパケットが集まる位置から
    % 抜けた時刻を推定して決める。
    TBall = mapTime(evB.t, tkB, pp.T, ver.stepThr, []);
    TAall = mapTime(evA.t, tkA, pp.T, ver.stepThr, []);
    tauA  = gapSplits(tkA, pp.T, TBall, ver.stepThr);
    tauB  = gapSplits(tkB, pp.T, TAall, ver.stepThr);
    mapA  = @(t) mapTime(t, tkA, pp.T, ver.stepThr, tauA);
    mapB  = @(t) mapTime(t, tkB, pp.T, ver.stepThr, tauB);

    startDelay = mapB(0) - mapA(0);     % other の開始が ref より何秒遅いか
    res.startDelaySec = startDelay;
    if startDelay >= 0
        fprintf('  %s は %s より %.3f 秒遅れて開始\n', cfg.otherTag, cfg.refTag, startDelay);
    else
        fprintf('  %s は %s より %.3f 秒早く開始\n', cfg.otherTag, cfg.refTag, -startDelay);
    end
    fprintf('  クロック差: %.2f ppm\n', ver.driftPpm);
    printVerification(ver, cfg, res);

    % --- 2 台がそろっている区間と、切り出し区間 (共通の時刻) ---
    a = max(mapA(0), mapB(0));
    b = min(mapA(D1), mapB(D2));
    overlap = b - a;
    res.overlapSec = overlap;

    dur = cfg.outputDuration;
    if overlap < dur
        warning('alignCSI:shortOverlap', ...
            '2 台がそろっている区間が %.1f 秒しかないため、%.1f 秒だけ出力します。', overlap, overlap);
        dur = overlap;
        winStart = a;
    elseif strcmpi(cfg.windowPosition, 'start')
        winStart = a;
    else
        winStart = a + (overlap - dur) / 2;
    end
    res.outputSec = dur;
    fprintf('  2 台がそろっている区間: 共通の時刻で %.2f〜%.2f s (%.1f 秒)\n', a, b, overlap);
    fprintf('  切り出し区間          : 共通の時刻で %.2f〜%.2f s (%.1f 秒) -> 出力では 0〜%.1f s\n', ...
        winStart, winStart + dur, dur, dur);

    % --- 5. 出力 ---
    [Aout, Bout, cnt] = buildOutputs(A, B, evA, evB, pp, mapA, mapB, winStart, dur, cfg);
    res.nHECommon    = cnt.nCommonHE;
    res.nHEOnlyRef   = cnt.nOnlyRefHE;
    res.nHEOnlyOther = cnt.nOnlyOtherHE;
    res.ratioRef     = cnt.nCommonHE / max(cnt.nUsableRefHE, 1);
    res.ratioOther   = cnt.nCommonHE / max(cnt.nUsableOtherHE, 1);
    printOutputCounts(cnt, cfg, res, dur);

    if cfg.commonPacketsOnly
        % 2 つのファイルで、行の対応と時刻が完全に一致していることを確かめる
        fmts = {'NonHT', 'HT', 'VHT', 'HE'};
        for f = 1:numel(fmts)
            tf = ['timeSec' fmts{f}];
            if isfield(Aout, tf) && isfield(Bout, tf) && ~isequal(Aout.(tf), Bout.(tf))
                error('alignCSI:notIdentical', '%s が 2 つのファイルで一致しません (内部の誤り)。', tf);
            end
        end
    end

    % --- 揃えたときの情報 (出力の csiMeta.alignment に残す) ---
    info = struct();
    info.version              = alignVersion();
    info.method               = 'packet-level matching of arrival times (same AP traffic)';
    info.referenceFile        = pr.refFile;
    info.otherFile            = pr.otherFile;
    info.commonPacketsOnly    = cfg.commonPacketsOnly;
    info.heLink               = cfg.heLink;
    info.timeAxis             = ['seconds from the window start; reference receiver clock ', ...
                                 'with sample gaps (overruns) restored'];
    info.windowStartCommonSec = winStart;   % 切り出し区間の先頭 (共通の時刻)
    info.outputDuration       = dur;
    info.otherStartDelaySec   = startDelay; % other の開始が ref より何秒遅いか
    info.coarseOffsetSec      = L0;
    info.confidence           = conf;
    info.clockDiffPpm         = ver.driftPpm;
    info.verification         = ver;
    info.matchedTimesRef      = tkA;        % 対応づけた全パケットの元の時刻 (検証用)
    info.matchedTimesOther    = tkB;
    info.matchedTimesCommon   = pp.T - winStart;
    info.lossTraceHE          = cnt.lossTraceHE;

    infoA = info; infoA.role = 'reference';
    infoB = info; infoB.role = 'other (mapped to the reference clock)';
    Aout.csiMeta.captureDuration = dur;
    Bout.csiMeta.captureDuration = dur;
    Aout.csiMeta.alignment = infoA;
    Bout.csiMeta.alignment = infoB;

    % --- 全フレームの記録 (IMU との時刻合わせ用) を、揃えた時間軸に直して引き継ぐ ---
    Aout = carryFrameLog(Aout, A, mapA, winStart);
    Bout = carryFrameLog(Bout, B, mapB, winStart);

    save(pr.refOut,   '-struct', 'Aout', '-v7.3');
    save(pr.otherOut, '-struct', 'Bout', '-v7.3');
    fprintf('  保存しました:\n    %s\n    %s\n', pr.refOut, pr.otherOut);
    res.status = 'OK';
end

function S = carryFrameLog(S, src, mapFn, winStart)
    % decodeIQ_HE.m の frameLog (全フレームの記録) を出力に引き継ぎ、
    % このファイルの時間軸での時刻 timeSecAligned を足す。
    % 変換は CSI と同じ mapA / mapB (サンプルの抜けも考慮) を使う。
    if ~isfield(src, 'frameLog') || isempty(src.frameLog) || ~isfield(src.frameLog, 'timeSec')
        return;
    end
    F = src.frameLog;
    F.timeSecAligned = reshape(mapFn(double(F.timeSec(:))) - winStart, [], 1);
    F.timeAxisNote = ['timeSec = this receiver''s original time (seconds from its capture start); ', ...
                      'timeSecAligned = time on the aligned axis of this file (0 = window start). ', ...
                      'Frames outside the window are kept.'];
    S.frameLog = F;
end

function res = failResult(res, msg)
    % 想定内の失敗。途中まで求めた値 (確信度など) は結果の一覧に残す。
    res.status  = 'FAIL';
    res.message = msg;
    fprintf(2, '\n  失敗しました: %s\n', msg);
end

function ev = collectEvents(S)
    % 対象 SSID の全パケットの (時刻, 種類) を時刻順に並べる。
    % 種類を分けておくのは、対応づけで取り違えを防ぐため (RTS の直後に HE の
    % データが来るなど、別のパケットが 0.1 ms 程度の近さで並ぶことがある)。
    % fmt / row は、元のファイルのどの形式の何行目か (1=Non-HT 2=HT 3=VHT 4=HE)。
    t    = zeros(0, 1);
    kind = cell(0, 1);
    fmt  = zeros(0, 1);
    row  = zeros(0, 1);
    legacy = {'NonHT', 'Non-HT'; 'HT', 'HT'; 'VHT', 'VHT'};
    for f = 1:size(legacy, 1)
        tt = double(fieldOr(S, ['timeSec' legacy{f, 1}], []));
        tt = tt(:);
        if isempty(tt)
            continue;
        end
        ft = fieldOr(S, ['frameType' legacy{f, 1}], {});
        if iscell(ft) && numel(ft) == numel(tt)
            kk = cellfun(@(x) [legacy{f, 2} ':' char(x)], ft(:), 'UniformOutput', false);
        else
            kk = repmat(legacy(f, 2), numel(tt), 1);
        end
        t = [t; tt]; kind = [kind; kk]; %#ok<AGROW>
        fmt = [fmt; f * ones(numel(tt), 1)]; row = [row; (1:numel(tt)).']; %#ok<AGROW>
    end

    tt = double(fieldOr(S, 'timeSecHE', []));
    tt = tt(:);
    if ~isempty(tt)
        ul = fieldOr(S, 'uplinkHE', []);
        if numel(ul) == numel(tt)
            kk = repmat({'HE-DL'}, numel(tt), 1);
            kk(logical(ul(:))) = {'HE-UL'};
        else
            kk = repmat({'HE'}, numel(tt), 1);
        end
        t = [t; tt]; kind = [kind; kk];
        fmt = [fmt; 4 * ones(numel(tt), 1)]; row = [row; (1:numel(tt)).'];
    end

    [t, order] = sort(t);
    ev = struct('t', t, 'kind', {kind(order)}, 'fmt', fmt(order), 'row', row(order));
end

function n = overrunCountOf(S)
    % キャプチャ時のオーバーラン回数 (記録が無ければ NaN)
    n = fieldOr(S.csiMeta, 'overrunCount', NaN);
    if isempty(n) || ~isnumeric(n)
        n = NaN;
    end
    n = double(n(1));
end

function D = captureLength(S)
    % キャプチャの実際の長さ [s]
    D = NaN;
    try
        D = double(S.csiMeta.captureMeta.totalSamples) / double(S.csiMeta.sampleRate);
    catch
    end
    if ~isfinite(D) || D <= 0
        D = double(S.csiMeta.captureDuration);
    end
end

function [evA, evB] = indexKinds(evA, evB)
    % 種類を番号にし、other は種類ごとのパケット番号の一覧を作っておく
    % (対応づけで何度も使うので、文字列の比較を毎回しないため)。
    [~, ~, id] = unique([evA.kind; evB.kind]);
    nA = numel(evA.t);
    evA.kid = id(1:nA);
    evB.kid = id(nA+1:end);
    nK = max([id; 0]);
    evB.byKind = cell(nK, 1);
    for q = 1:nK
        evB.byKind{q} = find(evB.kid == q);
    end
end

function co = coarseOffset(evA, evB, D1, M)
    % ref の 5 か所 (各 1 秒) について、other とのずれを ±M 秒の範囲で探す。
    % 同じ種類のパケット同士の時刻差を全部求め、幅 10 us に一番多く集まる
    % 時刻差を、その区間のずれとする。
    % 確信度 = その山の件数 / ±1 ms より外で 2 番目に大きい山の件数
    segLen = 1;
    fracs  = [0.3 0.4 0.5 0.6 0.7];
    bw     = 10e-6;
    co = struct('segCenter', [], 'offset', [], 'count', [], 'nRef', [], 'conf', []);
    for f = fracs
        s0  = max(0, D1 * f - segLen / 2);
        sel = find(evA.t >= s0 & evA.t < s0 + segLen);
        if numel(sel) < 50
            continue;
        end
        d = pairDiffs(evA, sel, evB, -M, M);
        if isempty(d)
            continue;
        end
        [loc, cnt] = diffPeaks(d, bw, 1e-3, 2);
        co.segCenter(end+1) = s0 + segLen / 2;
        co.offset(end+1)    = loc(1);
        co.count(end+1)     = cnt(1);
        co.nRef(end+1)      = numel(sel);
        co.conf(end+1)      = cnt(1) / max(cnt(2), 1);
    end
end

function [ia, ib] = fineMatch(evA, evB, D1, anchorT, L0)
    % 1 秒ごとにずれを追跡し、同じ種類のパケット同士を 1 対 1 で対応づける。
    % 戻り値は対応づけたパケットの番号 (ev の何番目か)。other の時刻順。
    winLen = 1;         % ずれを求め直す間隔 [s]
    tolAcc = 20e-6;     % その区間のずれからの許容幅 (ここでは広めに取り、
                        % verifyPairs で局所的なずれから ±数 us に絞る)

    nW   = max(1, ceil(max([D1; evA.t]) / winLen));
    wOf  = min(floor(evA.t / winLen) + 1, nW);   % ref の各パケットの区間番号
    offW = trackOffset(evA, evB, wOf, nW, winLen, anchorT, L0);
    [ia, ib] = matchPackets(evA, evB, offW, wOf, tolAcc);
end

function offW = trackOffset(evA, evB, wOf, nW, winLen, anchorT, L0)
    % 粗い位置合わせをした区間から前後へ 1 区間ずつ進み、前の区間のずれの
    % 近くで、その区間のずれを求める。2 台がそろっていない区間は NaN。
    offW = NaN(nW, 1);
    a = min(max(floor(anchorT / winLen) + 1, 1), nW);
    orders = {a:nW, a-1:-1:1};
    for k = 1:2
        p = L0;
        for w = orders{k}
            offW(w) = windowOffset(evA, find(wOf == w), evB, p);
            if ~isnan(offW(w))
                p = offW(w);
            end
        end
    end
end

function off = windowOffset(evA, sel, evB, p)
    % 1 区間分のずれ。まず前の区間のずれ p の ±20 ms で探し、見つからなければ
    % (大きなオーバーランの段差など) ±2 s に広げる。
    off = NaN;
    if numel(sel) < 5
        return;
    end
    thr = max(5, 0.1 * numel(sel));   % 2 台がそろっていない区間はここで落ちる
    for W = [20e-3, 2]
        d = pairDiffs(evA, sel, evB, p - W, p + W);
        if isempty(d)
            continue;
        end
        [loc, cnt] = diffPeaks(d, 10e-6, 1e-3, 2);
        if cnt(1) >= thr && cnt(1) >= 2 * cnt(2)
            off = loc(1);
            return;
        end
    end
end

function [ia, ib] = matchPackets(evA, evB, offW, wOf, tol)
    % 各区間のパケットを、その区間と前後の区間のずれ ±tol で対応づける。
    % オーバーランで段差がある区間では、段差の手前は前の区間と、後ろは
    % 次の区間と同じずれになるので、前後の区間のずれも候補に入れる。
    nW = numel(offW);
    parts = cell(nW, 3);
    for w = 1:nW
        nb   = [w, w - 1, w + 1];
        nb   = nb(nb >= 1 & nb <= nW);
        cand = offW(nb);
        cand = cand(~isnan(cand));
        sel  = find(wOf == w);
        if isempty(cand) || isempty(sel)
            continue;
        end
        IA = []; IB = []; R = [];
        for c = cand(:).'
            [d, ja, jb] = pairDiffs(evA, sel, evB, c - tol, c + tol);
            IA = [IA; ja]; IB = [IB; jb]; R = [R; abs(d - c)]; %#ok<AGROW>
        end
        parts(w, :) = {IA, IB, R};
    end
    ia = vertcat(parts{:, 1});
    ib = vertcat(parts{:, 2});
    r  = vertcat(parts{:, 3});
    if isempty(ia)
        ia = zeros(0, 1); ib = zeros(0, 1);
        return;
    end

    % 1 つのパケットに複数の相手が残った場合は、ずれに一番近いものを採る
    [~, ord] = sort(r);
    ia = ia(ord); ib = ib(ord);
    [~, first] = unique(ia, 'stable');  ia = ia(first); ib = ib(first);
    [~, first] = unique(ib, 'stable');  ia = ia(first); ib = ib(first);

    % 2 台の時刻の順番が入れ替わる対応は除く
    [~, ord] = sort(evB.t(ib));
    ia = ia(ord); ib = ib(ord);
    t1 = evA.t(ia);
    t2 = evB.t(ib);
    keep = [true; diff(t2) > 0 & diff(t1) > 0];
    ia = ia(keep); ib = ib(keep);

    % 前後 7 組の中央値から外れた対応を除く。孤立した取り違えだけが消え、
    % オーバーランによる本物の段差は (多数派が切り替わるだけなので) 残る。
    off  = evB.t(ib) - evA.t(ia);
    keep = abs(off - movmedian(off, 15)) < tol;
    ia = ia(keep); ib = ib(keep);
end

function [pp, ver] = verifyPairs(A, B, evA, evB, ia, ib, tol)
    % 対応づけた組を確かめて絞り込み、2 台共通の時刻を決める。
    %
    %  (a) 中身の照合: タイミングとは独立に、2 つのパケットの中身が矛盾しないか
    %  (b) 孤立した対応: ずれが前後の組と違う (段差で区切ると 5 組未満になる) もの
    %  (c) 残差: 前後 31 組の中央値 (局所的なずれ) から ±tol を超えるもの
    %  (d) 段差: 隣の組とのずれの差が stepThr を超える所。オーバーランで
    %      サンプルが抜けた所で、ずれ (other - ref) が増えれば ref、減れば
    %      other で抜けている。ref で抜けた分は共通の時刻に空白として戻す
    %  (e) 取り違えの余地: 同じ種類のパケットどうしの最短間隔
    stepThr = 5e-6;
    minSeg  = 5;
    ver = struct('tolUs', 1e6 * tol, 'stepThr', stepThr, 'nInitial', numel(ia));

    % (a)
    [okC, ver.content] = contentCheck(A, B, evA, evB, ia, ib);
    ia = ia(okC);  ib = ib(okC);

    % (b)
    ver.nIsolated = 0;
    for it = 1:10
        [s0, s1] = segmentsOf(evB.t(ib) - evA.t(ia), stepThr);
        short = (s1 - s0 + 1) < minSeg;
        if ~any(short)
            break;
        end
        drop = false(size(ia));
        for s = find(short).'
            drop(s0(s):s1(s)) = true;
        end
        ver.nIsolated = ver.nIsolated + nnz(drop);
        ia = ia(~drop);  ib = ib(~drop);
    end

    % (c)
    off = evB.t(ib) - evA.t(ia);
    [s0, s1] = segmentsOf(off, stepThr);
    resid = zeros(size(off));
    for s = 1:numel(s0)
        k = (s0(s):s1(s)).';
        resid(k) = off(k) - movmedian(off(k), 31);
    end
    okR = abs(resid) <= tol;
    ver.nRejectedTol = nnz(~okR);
    ia = ia(okR);  ib = ib(okR);  resid = resid(okR);
    ar = 1e6 * abs(resid);
    ver.residMedianUs = median(ar);
    ver.residP999Us   = prctileSimple(ar, 99.9);
    ver.residMaxUs    = max([ar; 0]);
    ver.nPairs        = numel(ia);

    % (d)
    off = evB.t(ib) - evA.t(ia);
    [s0, s1] = segmentsOf(off, stepThr);
    nS = numel(s0) - 1;
    stepSize = zeros(nS, 1);
    for s = 1:nS
        before = s0(s):s1(s);
        before = before(max(1, end - 9):end);
        after  = s0(s + 1):s1(s + 1);
        after  = after(1:min(10, end));
        stepSize(s) = median(off(after)) - median(off(before));
    end
    refGap = zeros(size(off));          % それまでに ref で抜けた時間の合計
    for s = 1:nS
        if stepSize(s) > 0
            refGap(s0(s + 1):end) = refGap(s0(s + 1):end) + stepSize(s);
        end
    end
    pp = struct('ia', ia, 'ib', ib, 'T', evA.t(ia) + refGap);

    tStep = reshape(pp.T(s0(2:end)), [], 1);      % 段差の直後の組の共通の時刻
    isRef = stepSize > 0;
    ver.stepRefSec        = tStep(isRef);
    ver.stepRefSizeSec    = stepSize(isRef);
    ver.stepOtherSec      = tStep(~isRef);
    ver.stepOtherSizeSec  = -stepSize(~isRef);

    % クロック差 = 段差を除いたずれの変化 / 時間
    if numel(off) >= 2
        span = evA.t(ia(end)) - evA.t(ia(1));
        ver.driftPpm = 1e6 * (off(end) - off(1) - sum(stepSize)) / max(span, eps);
    else
        ver.driftPpm = NaN;
    end

    % (e)
    [ver.minSpacingRefUs,   ver.minSpacingRefKind]   = minSameKindSpacing(evA, ia);
    [ver.minSpacingOtherUs, ver.minSpacingOtherKind] = minSameKindSpacing(evB, ib);
end

function [ok, c] = contentCheck(A, B, evA, evB, ia, ib)
    % タイミングとは独立に、対応づけた 2 つのパケットの中身が矛盾しないかを見る。
    %   HE   : BSS Color (HE-SIG-A, CRC で保護) が一致すること
    %   全形式: 両方で FCS が確認できた場合は、MAC のフレーム種別が一致すること
    %          (Non-HT などはフレーム種別ごとに対応づけているので常に一致する)
    % 上り/下りの区別は種類 (HE-DL / HE-UL) として対応づけの時点で揃えている。
    ok = true(size(ia));
    c  = struct('nColorChecked', 0, 'nColorMismatch', 0, ...
                'nTypeChecked', 0, 'nTypeMismatch', 0);
    fmts = {'NonHT', 'HT', 'VHT', 'HE'};
    for f = 1:numel(fmts)
        sel = find(evA.fmt(ia) == f);
        if isempty(sel)
            continue;
        end
        fn = fmts{f};
        rA = evA.row(ia(sel));
        rB = evB.row(ib(sel));
        nA = numel(fieldOr(A, ['timeSec' fn], []));
        nB = numel(fieldOr(B, ['timeSec' fn], []));

        if f == 4
            ca = fieldOr(A, 'bssColorHE', []);
            cb = fieldOr(B, 'bssColorHE', []);
            if numel(ca) == nA && numel(cb) == nB
                ca = double(ca(:));  cb = double(cb(:));
                bad = ca(rA) ~= cb(rB);
                c.nColorChecked  = c.nColorChecked + numel(sel);
                c.nColorMismatch = c.nColorMismatch + nnz(bad);
                ok(sel(bad)) = false;
            end
        end

        fa = fieldOr(A, ['fcs' fn], []);        fb = fieldOr(B, ['fcs' fn], []);
        ta = fieldOr(A, ['frameType' fn], {});  tb = fieldOr(B, ['frameType' fn], {});
        if numel(fa) == nA && numel(fb) == nB && iscell(ta) && iscell(tb) && ...
                numel(ta) == nA && numel(tb) == nB
            fa = logical(fa(:));  fb = logical(fb(:));
            ta = ta(:);  tb = tb(:);
            both = find(fa(rA) & fb(rB));
            bad  = ~strcmp(ta(rA(both)), tb(rB(both)));
            c.nTypeChecked  = c.nTypeChecked + numel(both);
            c.nTypeMismatch = c.nTypeMismatch + nnz(bad);
            ok(sel(both(bad))) = false;
        end
    end
end

function [s0, s1] = segmentsOf(off, thr)
    % ずれの並びを、隣どうしの差が thr を超える所 (段差) で区切る
    if isempty(off)
        s0 = zeros(0, 1);  s1 = zeros(0, 1);
        return;
    end
    brk = find(abs(diff(off(:))) > thr);
    s0 = [1; brk + 1];
    s1 = [brk; numel(off)];
end

function [gUs, kindName] = minSameKindSpacing(ev, idx)
    % 対応づけたパケットそれぞれについて、同じ受信機の同じ種類で一番近い
    % 別のパケットまでの間隔を求め、その最小値を返す。これが許容幅の 2 倍より
    % 十分大きければ、許容幅の中に相手の候補は 1 つしかない (取り違えようがない)。
    gUs = NaN;  kindName = '';
    if isempty(idx)
        return;
    end
    near = inf(numel(ev.t), 1);
    for q = unique(ev.kid(idx)).'
        k = find(ev.kid == q);
        if numel(k) < 2
            continue;
        end
        d = diff(ev.t(k));
        near(k) = min([inf; d], [d; inf]);
    end
    [g, i] = min(near(idx));
    gUs = 1e6 * g;
    kindName = ev.kind{idx(i)};
end

function p = prctileSimple(x, q)
    % prctile (Statistics Toolbox) を使わない百分位点
    x = sort(x(:));
    if isempty(x)
        p = NaN;
        return;
    end
    p = x(max(1, min(numel(x), ceil(q / 100 * numel(x)))));
end

function T = mapTime(t, tk, Tk, stepThr, tau)
    % 受信機の時刻 t を共通の時刻へ変換する。節点 (対応づけたパケット) の
    % 局所の時刻 tk と共通の時刻 Tk から、(共通 - 局所) を節点の間で直線補間
    % する。間に段差 (サンプルの抜け) がある 2 つの節点の間では、抜けた局所の
    % 時刻 tau(k) (gapSplits で推定) より前は前の節点、後は後の節点の値を使う。
    % tau が無い・NaN の区間は近い方の節点の値を使う。範囲の外は端の節点の値。
    sz  = size(t);
    t   = double(t(:));
    tk  = tk(:);
    off = Tk(:) - tk;
    n   = numel(tk);

    [ts, ord] = sort(t);
    j = zeros(size(t));
    j(ord) = firstIndex(tk, ts, true) - 1;   % tk(j) <= t となる最後の節点 (0 = 無し)

    o = zeros(size(t));
    o(j == 0) = off(1);
    o(j >= n) = off(n);
    mid = find(j >= 1 & j < n);
    if ~isempty(mid)
        k  = j(mid);
        w  = (t(mid) - tk(k)) ./ (tk(k + 1) - tk(k));
        o0 = off(k);
        o1 = off(k + 1);
        om = o0 + (o1 - o0) .* w;
        jump = abs(o1 - o0) > stepThr;
        before = w <= 0.5;
        if ~isempty(tau)
            tauK  = tau(k);
            known = ~isnan(tauK);
            before(known) = t(mid(known)) < tauK(known);
        end
        side = o1;
        side(before) = o0(before);
        om(jump) = side(jump);
        o(mid) = om;
    end
    T = reshape(t + o, sz);
end

function tau = gapSplits(tk, Tk, Tother, stepThr)
    % この受信機でサンプルが抜けた所 (節点の間で 共通 - 局所 が増える所) ごとに、
    % 抜けた局所の時刻 tau を推定する。抜けていた間 (共通の時刻で長さ D) は
    % この受信機には何も届かず、相手だけが受ける。そこで、2 つの節点の間で
    % 相手のパケット (共通の時刻 Tother) が長さ D の窓に一番多く入る位置を、
    % 抜けていた区間とする。相手のパケットが無ければ NaN (近い方の節点を使う)。
    tk  = tk(:);
    off = Tk(:) - tk;
    d   = diff(off);
    tau = NaN(numel(tk) - 1, 1);
    Tother = sort(Tother(:));
    for k = find(d > stepThr).'
        D  = d(k);
        g0 = Tk(k);                 % 抜けの始まり (共通の時刻) が取りうる範囲
        g1 = Tk(k + 1) - D;
        q  = Tother(Tother > Tk(k) & Tother < Tk(k + 1));
        if isempty(q) || g1 < g0
            continue;
        end
        cand = [g0; g1; q; q - D];
        cand = unique(cand(cand >= g0 & cand <= g1));
        score = zeros(size(cand));
        for c = 1:numel(cand)
            score(c) = nnz(q >= cand(c) & q <= cand(c) + D);
        end
        best = cand(score == max(score));
        g = (min(best) + max(best)) / 2;
        tau(k) = g - off(k);        % 抜けた局所の時刻 (= 共通の時刻 - 抜ける前のずれ)
    end
end

function [Aout, Bout, cnt] = buildOutputs(A, B, evA, evB, pp, mapA, mapB, winStart, dur, cfg)
    % 切り出し区間のパケットを、共通の時刻で並べた出力を作る。
    fmts = {'NonHT', 'HT', 'VHT', 'HE'};
    Aout = A;
    Bout = B;
    cnt = struct('nCommonHE', 0, 'nOnlyRefHE', 0, 'nOnlyOtherHE', 0, ...
        'nUsableRefHE', 0, 'nUsableOtherHE', 0, ...
        'nLinkExcludedRefHE', 0, 'nLinkExcludedOtherHE', 0, ...
        'nInvalidRef', 0, 'nInvalidOther', 0, 'nCommonAll', 0, ...
        'lossTraceHE', struct('timeSec', zeros(0, 1), 'inRef', false(0, 1), 'inOther', false(0, 1)));
    for f = 1:numel(fmts)
        fn = fmts{f};
        tf = ['timeSec' fn];
        if ~isfield(A, tf) || ~isfield(B, tf)
            % 片方のファイルにしか無い形式は、揃えようがないので空にする
            if isfield(A, tf)
                Aout = takeRows(Aout, A, fn, zeros(0, 1), zeros(0, 1));
            end
            if isfield(B, tf)
                Bout = takeRows(Bout, B, fn, zeros(0, 1), zeros(0, 1));
            end
            continue;
        end
        tA = double(A.(tf)(:));
        tB = double(B.(tf)(:));
        nA = numel(tA);
        nB = numel(tB);

        % 対応づけた組のうち、この形式のもの
        isF = evA.fmt(pp.ia) == f;
        rA  = evA.row(pp.ia(isF));
        rB  = evB.row(pp.ib(isF));
        Tp  = pp.T(isF);

        % 各パケットの共通の時刻 (対応づけたものには 2 台で同じ値を入れる)
        TA = mapA(tA);  TA(rA) = Tp;
        TB = mapB(tB);  TB(rB) = Tp;

        inA = TA >= winStart & TA < winStart + dur;
        inB = TB >= winStart & TB < winStart + dur;
        lkA = linkMask(A, fn, nA, cfg.heLink);
        lkB = linkMask(B, fn, nB, cfg.heLink);
        okA = csiValid(A, fn, nA);
        okB = csiValid(B, fn, nB);
        selA = inA & lkA;
        selB = inB & lkB;
        cnt.nInvalidRef   = cnt.nInvalidRef   + nnz(selA & ~okA);
        cnt.nInvalidOther = cnt.nInvalidOther + nnz(selB & ~okB);

        % 2 台に共通: 両方で区間内・同じ上り/下り・CSI が有効
        both = selA(rA) & selB(rB) & okA(rA) & okB(rB);
        [Tc, o] = sort(Tp(both));
        rAc = rA(both);  rAc = rAc(o);
        rBc = rB(both);  rBc = rBc(o);
        cnt.nCommonAll = cnt.nCommonAll + numel(Tc);

        isComA = false(nA, 1);  isComA(rAc) = true;
        isComB = false(nB, 1);  isComB(rBc) = true;
        onlyA = find(selA & okA & ~isComA);
        onlyB = find(selB & okB & ~isComB);
        [~, o] = sort(TA(onlyA));  onlyA = onlyA(o);
        [~, o] = sort(TB(onlyB));  onlyB = onlyB(o);

        if cfg.commonPacketsOnly
            Aout = takeRows(Aout, A, fn, rAc, Tc - winStart);
            Bout = takeRows(Bout, B, fn, rBc, Tc - winStart);
            Aout.(['pairIndex' fn]) = (1:numel(rAc)).';
            Bout.(['pairIndex' fn]) = (1:numel(rBc)).';
        else
            rowsA = find(selA & okA);  [~, o] = sort(TA(rowsA));  rowsA = rowsA(o);
            rowsB = find(selB & okB);  [~, o] = sort(TB(rowsB));  rowsB = rowsB(o);
            Aout = takeRows(Aout, A, fn, rowsA, TA(rowsA) - winStart);
            Bout = takeRows(Bout, B, fn, rowsB, TB(rowsB) - winStart);
            % 相手のファイルでの行番号 (相手が受けていなければ 0)
            posA = zeros(nA, 1);  posA(rowsA) = 1:numel(rowsA);
            posB = zeros(nB, 1);  posB(rowsB) = 1:numel(rowsB);
            partnerA = zeros(nA, 1);  partnerA(rAc) = rBc;
            partnerB = zeros(nB, 1);  partnerB(rBc) = rAc;
            Aout.(['pairIndex' fn]) = lookupPos(partnerA(rowsA), posB);
            Bout.(['pairIndex' fn]) = lookupPos(partnerB(rowsB), posA);
        end

        if f == 4
            cnt.nCommonHE    = numel(Tc);
            cnt.nOnlyRefHE   = numel(onlyA);
            cnt.nOnlyOtherHE = numel(onlyB);
            cnt.nUsableRefHE   = nnz(selA & okA);
            cnt.nUsableOtherHE = nnz(selB & okB);
            cnt.nLinkExcludedRefHE   = nnz(inA & okA & ~lkA);
            cnt.nLinkExcludedOtherHE = nnz(inB & okB & ~lkB);
            if cfg.commonPacketsOnly
                Aout.unpairedHE = takeRows(struct(), A, 'HE', onlyA, TA(onlyA) - winStart);
                Bout.unpairedHE = takeRows(struct(), B, 'HE', onlyB, TB(onlyB) - winStart);
            end
            % 実測の欠損パターン: 区間内のパケットを時刻順に並べ、各受信機が
            % 受けたか (CSI が有効か) を記録する
            tAll = [Tc; TA(onlyA); TB(onlyB)] - winStart;
            inR  = [true(numel(Tc), 1); true(numel(onlyA), 1); false(numel(onlyB), 1)];
            inO  = [true(numel(Tc), 1); false(numel(onlyA), 1); true(numel(onlyB), 1)];
            [tAll, o] = sort(tAll);
            cnt.lossTraceHE = struct('timeSec', tAll, 'inRef', inR(o), 'inOther', inO(o));
        end
    end
    Aout = setPrimary(Aout);
    Bout = setPrimary(Bout);
end

function S = takeRows(S, src, fn, rows, tCommon)
    % src の形式 fn の行 rows を、共通の時刻 tCommon と一緒に S へ入れる
    tf = ['timeSec' fn];
    tl = double(src.(tf)(:));
    n  = numel(tl);
    rows = rows(:);
    S.(['timeSecLocal' fn]) = tl(rows);
    S.(tf) = tCommon(:);
    perPacket = {['csi' fn], ['frameType' fn], ['fcs' fn]};
    if strcmp(fn, 'HE')
        perPacket = [perPacket, {'uplinkHE', 'bssColorHE'}];
    end
    for v = perPacket
        name = v{1};
        if ~isfield(src, name) || isempty(src.(name))
            continue;
        end
        x = src.(name);
        if isvector(x) && ~ischar(x) && numel(x) == n && size(x, 1) ~= n
            x = x(:);       % 行ベクトルで保存されていた場合 (1 行 = 1 パケットに直す)
        end
        if size(x, 1) == n
            S.(name) = x(rows, :);
        else
            warning('alignCSI:rowMismatch', ...
                '%s の行数 (%d) が %s (%d) と合わないため空にします。', ...
                name, size(x, 1), tf, n);
            S.(name) = x([], :);
        end
    end
end

function m = linkMask(S, fn, n, link)
    % HE の上り/下りの選択 (HE 以外は全部 true)
    m = true(n, 1);
    if ~strcmp(fn, 'HE') || strcmpi(link, 'both')
        return;
    end
    ul = fieldOr(S, 'uplinkHE', []);
    if numel(ul) ~= n
        warning('alignCSI:noUplinkFlag', ...
            'uplinkHE が無いため、HE の上り/下りを分けられません (全部使います)。');
        return;
    end
    ul = logical(ul(:));
    if strcmpi(link, 'uplink')
        m = ul;
    else
        m = ~ul;
    end
end

function ok = csiValid(S, fn, n)
    % CSI が使えるパケット (全部 0 でも NaN でもない)。後段の位相サニタイズは
    % 全部 0 のパケットを捨てるので、片方だけで消えて番号がずれないよう先に除く
    c = fieldOr(S, ['csi' fn], []);
    if size(c, 1) ~= n
        ok = true(n, 1);
        return;
    end
    ok = all(isfinite(c), 2) & any(c ~= 0, 2);
end

function idx = lookupPos(partnerRows, pos)
    idx = zeros(size(partnerRows));
    has = partnerRows > 0;
    idx(has) = pos(partnerRows(has));
end

function S = setPrimary(S)
    % ResultCSI.m 互換の「主」データを、主フォーマットで作り直す
    pf  = S.csiMeta.primaryFormat;
    key = strrep(pf, '-', '');
    if isfield(S, ['csi' key]) && isfield(S, ['timeSec' key])
        S.csi         = S.(['csi' key]);
        S.timeSec     = S.(['timeSec' key]);
        S.frameType   = fieldOr(S, ['frameType' key], {});
        S.fcsVerified = fieldOr(S, ['fcs' key], []);
        S.phyFormat   = repmat({pf}, numel(S.timeSec), 1);
        S.packetStartIndex = round(S.timeSec(:) * S.csiMeta.sampleRate);
    end
end

function [d, ia, ib] = pairDiffs(evA, selA, evB, lo, hi)
    % ref のパケット selA のそれぞれについて、同じ種類の other のパケットの
    % うち (other の時刻) - (ref の時刻) が [lo, hi] に入るものを全部挙げる。
    % d = その時刻差、ia / ib = ref / other のパケット番号
    d = zeros(0, 1); ia = zeros(0, 1); ib = zeros(0, 1);
    selA = selA(:);
    if isempty(selA)
        return;
    end
    wantIdx = (nargout > 1);
    kA = evA.kid(selA);
    parts = cell(0, 3);
    for q = unique(kA).'
        a = selA(kA == q);
        b = evB.byKind{q};
        if isempty(b)
            continue;
        end
        r = evA.t(a);
        % other は関係する範囲 [r(1)+lo, r(end)+hi] だけに先に絞る
        o  = evB.t(b);
        i0 = bisect(o, r(1) + lo, false);
        i1 = bisect(o, r(end) + hi, true) - 1;
        if i1 < i0
            continue;
        end
        b = b(i0:i1);
        o = o(i0:i1);
        jLo = firstIndex(o, r + lo, false);       % o >= r+lo となる最初
        jHi = firstIndex(o, r + hi, true) - 1;    % o <= r+hi となる最後
        n   = max(jHi - jLo + 1, 0);
        N   = sum(n);
        if N == 0
            continue;
        end
        % repelem は対象が 1 個 (スカラー) だと行ベクトルを返すので、
        % 必ず列ベクトルにそろえる (その種類が区間に 1 回だけのとき)
        rep  = reshape(repelem((1:numel(a)).', n), [], 1);
        base = reshape(repelem(cumsum(n) - n, n), [], 1);
        jj   = jLo(rep) + (1:N).' - 1 - base;
        if wantIdx
            parts(end+1, :) = {o(jj) - r(rep), a(rep), b(jj)}; %#ok<AGROW>
        else
            parts(end+1, :) = {o(jj) - r(rep), [], []}; %#ok<AGROW>
        end
    end
    if ~isempty(parts)
        d  = vertcat(parts{:, 1});
        ia = vertcat(parts{:, 2});
        ib = vertcat(parts{:, 3});
    end
end

function k = bisect(o, x, orEqual)
    % 昇順の o で、o(k) >= x (orEqual なら o(k) > x) となる最初の k
    % (なければ numel(o)+1)。x はスカラー。
    lo = 1; hi = numel(o) + 1;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if o(mid) < x || (orEqual && o(mid) == x)
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    k = lo;
end

function j = firstIndex(o, x, orEqual)
    % o と x は昇順。各 x(i) について、o(j) >= x(i) (orEqual なら o(j) > x(i))
    % となる最初の j (なければ numel(o)+1)。
    % o と x をまとめて並べ替え、各 x(i) より前に来た o の数を数える。
    % sort は同じ値の順序を保つので、同じ値のときに o と x のどちらを先に
    % 置くかで「>=」と「>」を切り替えられる。
    n = numel(o);
    m = numel(x);
    if orEqual
        v   = [o(:); x(:)];                 % 同じ値なら o が前 -> o <= x を数える
        isX = [false(n, 1); true(m, 1)];
    else
        v   = [x(:); o(:)];                 % 同じ値なら x が前 -> o <  x を数える
        isX = [true(m, 1); false(n, 1)];
    end
    [~, ord] = sort(v);
    posX = find(isX(ord));                  % x は昇順なので元の順に並ぶ
    j = reshape(posX - (1:m).' + 1, size(x));
end

function [loc, cnt] = diffPeaks(d, bw, guard, nMax)
    % 時刻差 d の中で、幅 bw に一番多く集まる位置を大きい順に nMax 個。
    % 2 つ目以降は、それまでの山から ±guard 以上離れたところから探す。
    loc = NaN(nMax, 1);
    cnt = zeros(nMax, 1);
    if isempty(d)
        return;
    end
    d = sort(d(:));
    c = firstIndex(d, d + bw, true) - (1:numel(d)).';   % [d(i), d(i)+bw] の件数
    for p = 1:nMax
        [cp, i1] = max(c);
        if cp <= 0
            break;
        end
        cnt(p) = cp;
        loc(p) = median(d(i1 : i1 + cp - 1));
        c(abs(d - loc(p)) < guard) = 0;
    end
end

function v = fieldOr(S, name, default)
    if isfield(S, name)
        v = S.(name);
    else
        v = default;
    end
end

function printVerification(ver, cfg, res)
    fprintf('  対応づけの検証 (許容幅 ±%g us):\n', ver.tolUs);
    fprintf('    時刻の一致     : 中央値 %.2f us / 99.9%% %.2f us / 最大 %.2f us (%d 組)\n', ...
        ver.residMedianUs, ver.residP999Us, ver.residMaxUs, ver.nPairs);
    fprintf('    除外した組     : 許容幅の外 %d / 孤立した対応 %d / 中身の不一致 %d (最初 %d 組)\n', ...
        ver.nRejectedTol, ver.nIsolated, ...
        ver.content.nColorMismatch + ver.content.nTypeMismatch, ver.nInitial);
    fprintf('    中身の照合     : BSS Color 不一致 %d / %d 組、フレーム種別 不一致 %d / %d 組 (両方で FCS 確認済みのもの)\n', ...
        ver.content.nColorMismatch, ver.content.nColorChecked, ...
        ver.content.nTypeMismatch, ver.content.nTypeChecked);
    fprintf('    取り違えの余地 : 同じ種類のパケットどうしの最短間隔 %s %.1f us (%s) / %s %.1f us (%s)\n', ...
        cfg.refTag, ver.minSpacingRefUs, ver.minSpacingRefKind, ...
        cfg.otherTag, ver.minSpacingOtherUs, ver.minSpacingOtherKind);
    margin = 2 * ver.tolUs;
    if min(ver.minSpacingRefUs, ver.minSpacingOtherUs) > 5 * margin
        fprintf('                     許容幅の 2 倍 (%g us) より十分大きく、どのパケットにも相手の候補は 1 つしかない\n', margin);
    else
        warning('alignCSI:closeKinds', ...
            ['同じ種類のパケットが %.1f us の近さで並んでいる所があります。', ...
             '取り違えを避けるため pairToleranceUs を小さくしてください。'], ...
            min(ver.minSpacingRefUs, ver.minSpacingOtherUs));
    end

    fprintf('    時刻の段差     : %s で %d 回%s / %s で %d 回%s\n', ...
        cfg.refTag, numel(ver.stepRefSec), msList(ver.stepRefSizeSec), ...
        cfg.otherTag, numel(ver.stepOtherSec), msList(ver.stepOtherSizeSec));
    if ~isempty(ver.stepRefSec)
        fprintf('                     %s で抜けた分は、共通の時刻に空白として戻した\n', cfg.refTag);
    end
    tooManyRef   = isfinite(res.overrunRef)   && numel(ver.stepRefSec)   > res.overrunRef;
    tooManyOther = isfinite(res.overrunOther) && numel(ver.stepOtherSec) > res.overrunOther;
    if tooManyRef || tooManyOther
        warning('alignCSI:stepOverrun', ...
            ['段差の数がキャプチャ時のオーバーラン回数より多くなっています。', ...
             '2 台で同時にサンプルが抜けた可能性があり、その段差の大きさは正しくありません。']);
    elseif isfinite(res.overrunRef) && isfinite(res.overrunOther)
        fprintf('                     どちらもオーバーラン回数 (%g 回 / %g 回) 以内で、段差はオーバーランで説明できる\n', ...
            res.overrunRef, res.overrunOther);
    end
end

function printOutputCounts(cnt, cfg, res, dur)
    fprintf('  出力 (HE %s、%.1f 秒):\n', linkName(cfg.heLink), dur);
    fprintf('    2 台に共通     : %d 件 (%s の %.1f%% / %s の %.1f%%)\n', ...
        cnt.nCommonHE, cfg.refTag, 100 * res.ratioRef, cfg.otherTag, 100 * res.ratioOther);
    if cfg.commonPacketsOnly
        fprintf('                     -> 2 つのファイルに同じ順番で保存 (i 番目どうしが同じパケット、timeSec* も一致)\n');
        fprintf('    片方だけ       : %s %d 件 / %s %d 件 -> unpairedHE と csiMeta.alignment.lossTraceHE に保存\n', ...
            cfg.refTag, cnt.nOnlyRefHE, cfg.otherTag, cnt.nOnlyOtherHE);
    else
        fprintf('    片方だけ       : %s %d 件 / %s %d 件 (相手の行番号は pairIndexHE、無ければ 0)\n', ...
            cfg.refTag, cnt.nOnlyRefHE, cfg.otherTag, cnt.nOnlyOtherHE);
    end
    if ~strcmpi(cfg.heLink, 'both')
        other = ternary(strcmpi(cfg.heLink, 'downlink'), 'uplink', 'downlink');
        fprintf('    HE %s (除外) : %s %d 件 / %s %d 件\n', linkName(other), ...
            cfg.refTag, cnt.nLinkExcludedRefHE, cfg.otherTag, cnt.nLinkExcludedOtherHE);
    end
    if cnt.nInvalidRef + cnt.nInvalidOther > 0
        fprintf('    CSI が無効 (除外): %s %d 件 / %s %d 件 (全部 0 または NaN)\n', ...
            cfg.refTag, cnt.nInvalidRef, cfg.otherTag, cnt.nInvalidOther);
    end
    if min(res.ratioRef, res.ratioOther) < 0.8
        warning('alignCSI:lowCommon', ...
            ['2 台に共通の HE パケットが少なめです (%.0f%% / %.0f%%)。', ...
             '2 台の受信状況がかなり違う可能性があります。'], ...
            100 * res.ratioRef, 100 * res.ratioOther);
    end
end

function s = msList(sizes)
    if isempty(sizes)
        s = '';
        return;
    end
    parts = arrayfun(@(x) sprintf('%.2f', 1e3 * x), sizes, 'UniformOutput', false);
    s = sprintf(' (%s ms)', strjoin(parts(:).', ', '));
end

function s = linkName(link)
    switch lower(link)
        case 'downlink'
            s = '下り (AP->クライアント)';
        case 'uplink'
            s = '上り (クライアント->AP)';
        otherwise
            s = '上り+下り';
    end
end

function printAlignSummary(results)
    fprintf('\n=========================================================================\n');
    fprintf(' 時間軸の揃え方の結果\n');
    fprintf('=========================================================================\n');
    for k = 1:numel(results)
        r = results(k);
        switch r.status
            case 'OK'
                fprintf('  OK    %s\n', r.label);
                fprintf(['          開始のずれ %+.3f s, 対応 %d 組 (残差 最大 %.2f us, 最短間隔 %.0f us), ', ...
                         '段差 %d / %d, 出力 %.1f s\n'], ...
                    r.startDelaySec, r.nPairs, r.residMaxUs, r.minSpacingUs, ...
                    r.stepsRef, r.stepsOther, r.outputSec);
                fprintf('          HE 共通 %d 件 (%.1f%% / %.1f%%), 片方だけ %d / %d 件\n', ...
                    r.nHECommon, 100 * r.ratioRef, 100 * r.ratioOther, r.nHEOnlyRef, r.nHEOnlyOther);
            case 'SKIP'
                fprintf('  SKIP  %s (出力済み)\n', r.label);
            otherwise
                fprintf('  FAIL  %s\n          %s\n', r.label, r.message);
        end
    end
end

function writeAlignCsv(csvFile, results)
    fid = fopen(csvFile, 'w', 'n', 'UTF-8');
    if fid == -1
        return;
    end
    fprintf(fid, '%s', char(65279));   % Excel 向けの BOM
    fprintf(fid, ['pair,status,start_delay_s,confidence,matched_pairs,resid_max_us,', ...
                  'min_spacing_us,rejected,content_mismatch,steps_ref,steps_other,', ...
                  'overrun_ref,overrun_other,clock_diff_ppm,overlap_s,output_s,', ...
                  'HE_common,HE_only_ref,HE_only_other,ratio_ref,ratio_other,message\n']);
    for k = 1:numel(results)
        r = results(k);
        fprintf(fid, ['"%s",%s,%.4f,%.2f,%d,%.3f,%.1f,%d,%d,%d,%d,%g,%g,%.3f,%.2f,%.2f,', ...
                      '%d,%d,%d,%.4f,%.4f,"%s"\n'], ...
            r.label, r.status, r.startDelaySec, r.confidence, r.nPairs, r.residMaxUs, ...
            r.minSpacingUs, r.nRejected, r.nMismatch, r.stepsRef, r.stepsOther, ...
            r.overrunRef, r.overrunOther, r.driftPpm, r.overlapSec, r.outputSec, ...
            r.nHECommon, r.nHEOnlyRef, r.nHEOnlyOther, r.ratioRef, r.ratioOther, ...
            strrep(regexprep(r.message, '\s+', ' '), '"', '""'));
    end
    fclose(fid);
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
