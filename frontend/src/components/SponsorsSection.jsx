import React from 'react';

/**
 * Supporters & Sponsors component.
 * Renders the framed supporters & sponsors section.
 * Defaults to blank/placeholder state as requested.
 */
export default function SponsorsSection({
  platinumSponsors = [],
  silverSponsors = [],
  eventSponsors = []
}) {
  const isBlank = platinumSponsors.length === 0 && silverSponsors.length === 0 && eventSponsors.length === 0;

  return (
    <div className="card sponsors-card-frame" style={{
      background: 'rgba(20, 10, 15, 0.65)',
      backdropFilter: 'blur(16px)',
      border: '1px solid rgba(212, 175, 55, 0.25)',
      borderRadius: '16px',
      padding: '32px 24px',
      position: 'relative',
      overflow: 'hidden',
      marginTop: '24px'
    }}>
      {/* Decorative Ornaments in corners */}
      <div className="sponsors-corner sponsors-corner--tl" />
      <div className="sponsors-corner sponsors-corner--tr" />
      <div className="sponsors-corner sponsors-corner--bl" />
      <div className="sponsors-corner sponsors-corner--br" />

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '28px' }}>
        <h2 style={{
          color: '#F3E5AB',
          fontSize: '1.4rem',
          fontWeight: '800',
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          margin: 0,
          fontFamily: 'serif'
        }}>
          Supporters & Sponsors
        </h2>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          marginTop: '8px',
          color: '#D4AF37',
          opacity: 0.85
        }}>
          <span>◇</span>
          <div style={{ width: '40px', height: '1px', background: 'linear-gradient(90deg, transparent, #D4AF37)' }} />
          <span>◆</span>
          <div style={{ width: '40px', height: '1px', background: 'linear-gradient(90deg, #D4AF37, transparent)' }} />
          <span>◇</span>
        </div>
      </div>

      {/* Sponsors Content */}
      {isBlank ? (
        <div style={{
          background: 'rgba(45, 12, 20, 0.5)',
          border: '1px dashed rgba(212, 175, 55, 0.3)',
          borderRadius: '12px',
          padding: '40px 20px',
          textAlign: 'center',
          color: 'rgba(243, 229, 171, 0.7)'
        }}>
          <p style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600, letterSpacing: '0.05em' }}>
            Sponsor Tiers & Partners (Blank for now)
          </p>
          <span style={{ fontSize: '0.75rem', opacity: 0.6, marginTop: '4px', display: 'block' }}>
            Details will be populated upon announcement.
          </span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Platinum Tier */}
          <div className="sponsor-tier-box">
            <div className="sponsor-tier-title">PLATINUM SPONSOR</div>
            <div className="sponsor-tier-divider" />
            <div className="sponsor-tier-items">
              {platinumSponsors.map((name, i) => (
                <div key={i} className="sponsor-pill">{name}</div>
              ))}
            </div>
          </div>

          {/* Silver Tier */}
          <div className="sponsor-tier-box">
            <div className="sponsor-tier-title">SILVER SPONSOR</div>
            <div className="sponsor-tier-divider" />
            <div className="sponsor-tier-items">
              {silverSponsors.map((name, i) => (
                <div key={i} className="sponsor-pill">{name}</div>
              ))}
            </div>
          </div>

          {/* Event Tier */}
          <div className="sponsor-tier-box">
            <div className="sponsor-tier-title">EVENT SPONSOR</div>
            <div className="sponsor-tier-divider" />
            <div className="sponsor-tier-items">
              {eventSponsors.map((name, i) => (
                <div key={i} className="sponsor-pill">{name}</div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
