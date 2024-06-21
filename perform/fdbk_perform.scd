// Player


s.boot;
("fdbk_synthdef.scd").loadRelative;


// The two synthesizer
(
~leftSynth = Synth.new(\fdbk, [\pan, -1, \input, 0]);
~rightSynth = Synth.new(\fdbk, [\pan, 1, \input, 1]);
);


// The OSC receivers
(
OSCdef('\channel0', {
	arg msg;

	msg.postln;

	~leftSynth.set(
		\fund, msg[1],
	    \add_amp, msg[2],
		\blip_amp, msg[3],
		\fm_amp, msg[4],
		\ratio, msg[5],
		\index, msg[6],
		\pnoise_amp, msg[7],
		\bnoise_amp, msg[8],
		\lfp_freq, msg[9],
	    \hfp_freq, msg[10]);
}, '/channel_0')
);

(
OSCdef('\channel1', {
	arg msg;

	msg.postln;

	~rightSynth.set(
		\fund, msg[1],
	    \add_amp, msg[2],
		\blip_amp, msg[3],
		\fm_amp, msg[4],
		\ratio, msg[5],
		\index, msg[6],
		\pnoise_amp, msg[7],
		\bnoise_amp, msg[8],
		\lfp_freq, msg[9],
	    \hfp_freq, msg[10]);
}, '/channel_1')
)




