// Player

Server.default.options.inDevice_("M4");
Server.default.options.outDevice_("Built-in Output");

s.boot;
("fdbk_synthdef.scd").loadRelative;


// The two synthesizers
(
~leftSynth = Synth.new(\fdbk, [\in_chan, 0, \out_chan, 0, \dry_env, 0, \verb_mix, 0.5]);
~rightSynth = Synth.new(\fdbk, [\in_chan, 1, \out_chan, 1, \dry_env, 0, \verb_mix, 0.5]);
);


// The OSC receivers
(
OSCdef('\channel0', {
	arg msg;

	msg.postln;

	~leftSynth.set(
		\fund, msg[1],
	    \amp_ratio, msg[2],
		\fm_ratio, msg[3],
		\fm_index, msg[4],
		\pnoise_amp, msg[5],
		\lpf_freq, msg[6],
		\hpf_freq, msg[7],
		\osc_fm_ratio, msg[8]);
}, '/channel_0')
);

(
OSCdef('\channel1', {
	arg msg;

	msg.postln;

	~rightSynth.set(
		\fund, msg[1],
	    \amp_ratio, msg[2],
		\fm_ratio, msg[3],
		\fm_index, msg[4],
		\pnoise_amp, msg[5],
		\lpf_freq, msg[6],
		\hpf_freq, msg[7],
		\osc_fm_ratio, msg[8]);
}, '/channel_1')
)


~leftSynth.set(\lag_t, 0.5, \mix, 0.4)
~rightSynth.set(\lag_t, 0.5, \mix, 1)
