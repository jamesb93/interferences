(
var folder;
~src =  Array.new;

folder = PathName("/home/james/Cloud/Projects/ElectroMagnetic/segments/2_ExplodeAudio");
folder.entries.do({
    arg path;
    if (path.extension == "wav") {
        ~src = ~src.add(Buffer.read(s, path.fullPath));
        postln("Loading " ++ path.fileName);
    };
});
postln("---- Samples Loadead ----")
)

(
SynthDef.new(\reader, {
    arg buf, amp=1;
    var sig;
    sig = PlayBuf.ar(2, buf, doneAction:2);
    sig = sig * amp;
    Out.ar([0, 1], sig);
}).add;
)

Synth.new(\reader, [\buf, ~src[9]]);
s.freeAll;

{SinOsc.ar(440)}.play;

